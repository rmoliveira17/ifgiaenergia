# xgb_optuna_fix.py
# Requisitos:
#   pip install pandas numpy xgboost joblib openpyxl optuna

import os
import json
import joblib  # só se quiser salvar objetos auxiliares
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.integration import XGBoostPruningCallback

# ====================== CONFIG ======================
ARQ = "Modelo Final.xlsx"   # caminho do seu Excel
ABA = None                  # nome da aba (None = primeira)

TEST_SIZE   = 0.20          # fração mais recente para TESTE
VALID_FRAC  = 0.10          # fração final do TREINO para VALIDAÇÃO
RANDOM_STATE = 42

# Optuna
N_TRIALS   = 40
TIMEOUT    = None
STUDY_NAME = "xgb_fv_optuna"
STORAGE    = "sqlite:///xgb_fv_optuna.db"      # ex.: "sqlite:///xgb_fv_optuna.db" p/ persistir
# ====================================================

def _ensure_cols(df, cols, alt_map=None, what="feature"):
    alt_map = alt_map or {}
    out = []
    for c in cols:
        if c in df.columns:
            out.append(c)
        elif c in alt_map and alt_map[c] in df.columns:
            out.append(alt_map[c])
        else:
            raise ValueError(f"Coluna {what} ausente: '{c}'")
    return out

def _to_numeric_and_impute(df_train, df_to_fix, strategy="median"):
    df_train_num = df_train.apply(pd.to_numeric, errors="coerce")
    df_to_fix_num = df_to_fix.apply(pd.to_numeric, errors="coerce")
    stats = df_train_num.median(numeric_only=True) if strategy == "median" else df_train_num.mean(numeric_only=True)
    df_train_num = df_train_num.fillna(stats).fillna(0.0)
    df_to_fix_num = df_to_fix_num.fillna(stats).fillna(0.0)
    return df_train_num, df_to_fix_num

def _load_and_prepare():
    if not os.path.exists(ARQ):
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ}")
    df = pd.read_excel(ARQ, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    mes_col = "Mês" if "Mês" in df.columns else ("Mes" if "Mes" in df.columns else None)
    if not {"Cidade", "Ano"}.issubset(df.columns) or mes_col is None:
        raise ValueError("Preciso de 'Cidade', 'Ano' e 'Mês/Mes'.")

    df = df.sort_values(['Cidade', 'Ano', mes_col])

    # alvo acumulado por cidade (igual ao seu)
    agrupado_por_cidade_ano = df.groupby(by=['Cidade'])['Produção de Energia FV'].cumsum()
    df['Produção FV acumulada'] = agrupado_por_cidade_ano

    possiveis_targets = ["Produção FV acumulada", "Produção", "Producao de Energia"]
    target = next((t for t in possiveis_targets if t in df.columns), None)
    if target is None:
        raise ValueError("Coluna alvo não encontrada.")

    features_orig = ["Produção de Energia FV", "Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
    features = _ensure_cols(df, features_orig, alt_map={"Área": "Area"}, what="de feature")

    df["__DATA__"] = pd.to_datetime(dict(
        year=df["Ano"].astype(int),
        month=df[mes_col].astype(int),
        day=1
    ))
    df = df[~pd.isna(df[target])].sort_values("__DATA__").reset_index(drop=True)

    return df, target, features, mes_col

def _temporal_splits(df, features, target):
    X_all = df[features].copy()
    y_all = df[target].astype(float).values

    n = len(df)
    n_test = max(1, int(np.floor(TEST_SIZE * n)))
    cut_test = n - n_test

    X_train_full = X_all.iloc[:cut_test, :].copy()
    y_train_full = y_all[:cut_test]
    X_test = X_all.iloc[cut_test:, :].copy()
    y_test = y_all[cut_test:]
    datas_test = df["__DATA__"].iloc[cut_test:]

    n_tr = len(X_train_full)
    n_val = max(1, int(np.floor(VALID_FRAC * n_tr)))
    cut_val = n_tr - n_val

    X_tr = X_train_full.iloc[:cut_val, :].copy()
    y_tr = y_train_full[:cut_val]
    X_val = X_train_full.iloc[cut_val:, :].copy()
    y_val = y_train_full[cut_val:]

    return X_tr, y_tr, X_val, y_val, X_test, y_test, datas_test

def _objective_factory(X_tr_df, y_tr, X_val_df, y_val, X_test_df):
    # Usamos xgboost.train (DMatrix) para contornar ausência de callbacks no sklearn API.
    def objective(trial: optuna.Trial):
        # ----- espaço de hiperparâmetros -----
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),  # reg_lambda
            "alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),    # reg_alpha
            "eta": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),   # learning_rate
            "tree_method": "hist",
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "seed": RANDOM_STATE,
        }
        num_boost_round = trial.suggest_int("n_estimators", 500, 4000)
        es_rounds = trial.suggest_int("early_stopping_rounds", 30, 200)

        # ----- dados numéricos + imputação -----
        X_tr, X_val = _to_numeric_and_impute(X_tr_df.copy(), X_val_df.copy(), strategy="median")
        _,    X_te  = _to_numeric_and_impute(X_tr.copy(),    X_test_df.copy(), strategy="median")

        dtrain = xgb.DMatrix(X_tr.values.astype(np.float32), label=y_tr)
        dvalid = xgb.DMatrix(X_val.values.astype(np.float32), label=y_val)

        # ----- pruning callback (na API train) -----
        pruning_cb = XGBoostPruningCallback(trial, "validation-rmse")

        evals = [(dtrain, "train"), (dvalid, "validation")]
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=es_rounds,
            callbacks=[pruning_cb],
            verbose_eval=False
        )

        # RMSE de validação e melhor iteração
        val_rmse = float(booster.best_score)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))

        return val_rmse
    return objective

def _train_final_and_save(best_params, df, features, mes_col,
                          X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test):
    # TREINO+VAL
    X_tr, X_val = _to_numeric_and_impute(X_tr_df.copy(), X_val_df.copy(), strategy="median")
    X_trv_df = pd.concat([X_tr, X_val], axis=0)
    y_trv = np.concatenate([y_tr, y_val], axis=0)

    # TESTE imputado com stats do treino+val
    _, X_te = _to_numeric_and_impute(X_trv_df.copy(), X_test_df.copy(), strategy="median")

    dtrain = xgb.DMatrix(X_trv_df.values.astype(np.float32), label=y_trv)
    dtest  = xgb.DMatrix(X_te.values.astype(np.float32),     label=y_test)

    # Mapeia parâmetros Optuna -> XGB train
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": best_params["max_depth"],
        "min_child_weight": best_params["min_child_weight"],
        "gamma": best_params["gamma"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "lambda": best_params["reg_lambda"],
        "alpha": best_params["reg_alpha"],
        "eta": best_params["learning_rate"],
        "tree_method": "hist",
        "max_bin": best_params["max_bin"],
        "seed": RANDOM_STATE,
    }

    num_boost_round = int(best_params.get("best_iteration", best_params["n_estimators"])) + 1

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train")],
        verbose_eval=False
    )

    # Predição no TESTE
    # Usa melhor limite de árvores; para compatibilidade geral, use best_ntree_limit (se disponível)
    if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
        yhat = booster.predict(dtest, iteration_range=(0, booster.best_ntree_limit))
    else:
        yhat = booster.predict(dtest)

    rmse = float(np.sqrt(mean_squared_error(y_test, yhat)))
    mae  = float(mean_absolute_error(y_test, yhat))
    r2   = float(r2_score(y_test, yhat))

    outdir = "artefatos_xgb_optuna"
    os.makedirs(outdir, exist_ok=True)

    n_test = len(y_test)
    cols_saida = []
    if "Cidade" in df.columns:
        cols_saida.append("Cidade")
    cols_saida += ["Ano", mes_col]

    saida = df.iloc[-n_test:, :][cols_saida].copy()
    saida["y_true"] = y_test
    saida["yhat_xgb"] = yhat
    saida["data_ref"] = datas_test.values
    saida.to_csv(f"{outdir}/predicoes_teste.csv", index=False, encoding="utf-8")

    # Salva o modelo como JSON (Booster)
    booster.save_model(f"{outdir}/modelo_xgb.json")

    with open(f"{outdir}/metricas.json", "w", encoding="utf-8") as f:
        json.dump({"RMSE": rmse, "MAE": mae, "R2": r2}, f, ensure_ascii=False, indent=2)

    with open(f"{outdir}/best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print("\nMétricas no TESTE (mais recente):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print("\nArquivos gerados em ./artefatos_xgb_optuna:")
    print("- predicoes_teste.csv")
    print("- modelo_xgb.json")
    print("- metricas.json")
    print("- best_params.json")

def main():
    df, target, features, mes_col = _load_and_prepare()
    X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test = _temporal_splits(df, features, target)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=(STORAGE is not None)
    )
    objective = _objective_factory(X_tr_df, y_tr, X_val_df, y_val, X_test_df)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)

    print("\n=== RESULTADOS OPTUNA ===")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  Val RMSE: {study.best_value:.6f}")
    print("  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    best_params = study.best_trial.params.copy()
    best_it = study.best_trial.user_attrs.get("best_iteration", None)
    if best_it is not None:
        best_params["best_iteration"] = int(best_it)

    _train_final_and_save(
        best_params, df, features, mes_col,
        X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test
    )

    print("\nTop-5 trials por RMSE de validação:")
    top5 = sorted(study.trials, key=lambda t: t.value)[:5]
    for t in top5:
        print(f"  Trial #{t.number:>3} | val_RMSE={t.value:.6f} | params={t.params}")

if __name__ == "__main__":
    main()

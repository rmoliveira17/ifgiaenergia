# mlp_optuna.py
# Requisitos:
#   pip install pandas numpy tensorflow openpyxl optuna

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.integration import TFKerasPruningCallback

# ====================== CONFIG ======================
ARQ = "Modelo Final.xlsx"   # caminho do seu Excel
ABA = None                  # nome da aba (None = primeira)

TEST_SIZE   = 0.20          # fração mais recente para TESTE
VALID_FRAC  = 0.10          # fração final do TREINO para VALIDAÇÃO
RANDOM_SEED = 42

# Optuna
N_TRIALS = 30               # número de trials (ajuste conforme tempo disponível)
TIMEOUT  = None             # em segundos (ex.: 1800 para 30 min). Use None se quiser ignorar.
STUDY_NAME = "mlp_fv_optuna"
STORAGE = "sqlite:///mlp_fv_optuna.db"           # ex.: "sqlite:///mlp_fv_optuna.db" para persistir; None mantém em memória
# ====================================================

def ensure_cols(df, cols, alt_map=None, what="feature"):
    alt_map = alt_map or {}
    resolved = []
    for c in cols:
        if c in df.columns:
            resolved.append(c)
        elif c in alt_map and alt_map[c] in df.columns:
            resolved.append(alt_map[c])
        else:
            raise ValueError(f"Coluna {what} ausente: '{c}'")
    return resolved

def to_numeric_and_impute(train_df, df_to_fix, strategy="median"):
    train_num = train_df.apply(pd.to_numeric, errors="coerce")
    fix_num   = df_to_fix.apply(pd.to_numeric, errors="coerce")
    stats = train_num.median(numeric_only=True) if strategy == "median" else train_num.mean(numeric_only=True)
    train_num = train_num.fillna(stats).fillna(0.0)
    fix_num   = fix_num.fillna(stats).fillna(0.0)
    return train_num, fix_num, stats

def zscore_fit(X_train_df):
    mu  = X_train_df.mean(axis=0).values.astype(np.float32)
    sd  = X_train_df.std(axis=0, ddof=0).values.astype(np.float32)
    sd_safe = np.where(sd == 0, 1.0, sd).astype(np.float32)
    return mu, sd_safe

def zscore_apply(df_part, mu, sd):
    arr = df_part.values.astype(np.float32)
    return (arr - mu) / sd

def build_mlp(input_dim, lr=1e-3, dprob=0.1, h1=128, h2=64):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(h1, activation="relu"),
        layers.Dropout(dprob),
        layers.Dense(h2, activation="relu"),
        layers.Dropout(dprob),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def load_and_prepare():
    if not os.path.exists(ARQ):
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ}")

    df = pd.read_excel(ARQ, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # Ordenação temporal por cidade
    if not {"Cidade", "Ano"}.issubset(df.columns):
        raise ValueError("Preciso de colunas 'Cidade' e 'Ano' para ordenar.")
    mes_col = "Mês" if "Mês" in df.columns else ("Mes" if "Mes" in df.columns else None)
    if mes_col is None:
        raise ValueError("Preciso de 'Mês/Mes'.")

    df = df.sort_values(['Cidade', 'Ano', mes_col])

    # Target (mantém sua lógica de acumulado)
    agrupado_por_cidade_ano = df.groupby(by=['Cidade'])['Produção de Energia FV'].cumsum()
    df['Produção FV acumulada'] = agrupado_por_cidade_ano

    possiveis_targets = ["Produção FV acumulada", "Produção", "Producao de Energia"]
    target = next((t for t in possiveis_targets if t in df.columns), None)
    if target is None:
        raise ValueError("Coluna alvo não encontrada (ex.: 'Produção de Energia FV' ou 'Produção FV acumulada').")

    # Features
    features_orig = ["Produção de Energia FV", "Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
    features = ensure_cols(df, features_orig, alt_map={"Área": "Area"}, what="de feature")

    # Coluna de data para split temporal
    df["__DATA__"] = pd.to_datetime(dict(
        year=df["Ano"].astype(int),
        month=df[mes_col].astype(int),
        day=1
    ))
    df = df[~pd.isna(df[target])].sort_values("__DATA__").reset_index(drop=True)

    return df, target, features, mes_col

def temporal_splits(df, features, target):
    X_all = df[features].copy()
    y_all = df[target].astype(float).values

    n = len(df)
    n_test = max(1, int(np.floor(TEST_SIZE * n)))
    cut_test = n - n_test

    X_train_full = X_all.iloc[:cut_test, :].copy()
    y_train_full = y_all[:cut_test]
    X_test_df    = X_all.iloc[cut_test:, :].copy()
    y_test       = y_all[cut_test:]
    datas_test   = df["__DATA__"].iloc[cut_test:]

    # Split interno do treino -> validação temporal
    n_tr = len(X_train_full)
    n_val = max(1, int(np.floor(VALID_FRAC * n_tr)))
    cut_val = n_tr - n_val

    X_tr_df = X_train_full.iloc[:cut_val, :].copy()
    y_tr    = y_train_full[:cut_val]
    X_val_df= X_train_full.iloc[cut_val:, :].copy()
    y_val   = y_train_full[cut_val:]

    return (X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test, X_train_full, y_train_full)

def make_scaled_datasets(X_tr_df, X_val_df, X_test_df):
    # Imputação e padronização com stats do treino
    X_tr_df, X_val_df, medianas = to_numeric_and_impute(X_tr_df, X_val_df, strategy="median")
    _,      X_test_df, _        = to_numeric_and_impute(X_tr_df, X_test_df, strategy="median")
    mu, sd = zscore_fit(X_tr_df)
    X_tr = zscore_apply(X_tr_df,  mu, sd)
    X_val = zscore_apply(X_val_df, mu, sd)
    X_te  = zscore_apply(X_test_df, mu, sd)
    return X_tr, X_val, X_te, medianas, mu, sd

def set_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def objective_factory(X_tr_df, y_tr, X_val_df, y_val, X_test_df, features):
    # Pré-processamento fixo por trial (stats regeneradas a cada trial para consistência)
    def objective(trial: optuna.Trial):
        seed = RANDOM_SEED  # fixe, ou use: trial.suggest_int("seed", 1, 10_000)
        set_seeds(seed)

        # Espaço de busca
        lr       = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dprob    = trial.suggest_float("dropout", 0.0, 0.5)
        h1       = trial.suggest_categorical("h1", [64, 128, 256, 384, 512])
        h2       = trial.suggest_categorical("h2", [32, 64, 128, 256])
        batch_sz = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        epochs   = trial.suggest_int("epochs", 80, 400)
        patience = trial.suggest_int("patience", 10, 40)

        # Recalcula imputação e scaler com base SOMENTE em X_tr_df
        X_tr, X_val, X_te, medianas, mu, sd = make_scaled_datasets(X_tr_df.copy(), X_val_df.copy(), X_test_df.copy())

        # Modelo
        model = build_mlp(
            input_dim=X_tr.shape[1],
            lr=lr, dprob=dprob, h1=h1, h2=h2
        )

        # Callbacks: EarlyStopping + Pruning
        early = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        prune_cb = TFKerasPruningCallback(trial, "val_loss")

        history = model.fit(
            X_tr, y_tr.astype(np.float32),
            validation_data=(X_val, y_val.astype(np.float32)),
            epochs=epochs,
            batch_size=batch_sz,
            verbose=0,
            callbacks=[early, prune_cb]
        )

        # Usa RMSE de validação como objetivo
        y_val_pred = model.predict(X_val, verbose=0).ravel()
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        # Reporta para Optuna (para gráficos e história)
        trial.set_user_attr("val_rmse", float(rmse_val))
        trial.set_user_attr("epochs_run", len(history.history["loss"]))

        return rmse_val  # minimizar
    return objective

def train_final_and_save(best_params, df, features, target, mes_col,
                         X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test):
    # Re-treina em TREINO+VAL com os melhores hiperparâmetros e avalia no TESTE
    X_train_full_df = pd.concat([X_tr_df, X_val_df], axis=0)
    y_train_full    = np.concatenate([y_tr, y_val], axis=0)

    # Imputação/scaler com base no TREINO+VAL
    X_trv_df, X_te_df, medianas = to_numeric_and_impute(X_train_full_df.copy(), X_test_df.copy(), strategy="median")
    mu, sd = zscore_fit(X_trv_df)
    X_trv = zscore_apply(X_trv_df, mu, sd)
    X_te  = zscore_apply(X_te_df,  mu, sd)

    # Modelo final
    model = build_mlp(
        input_dim=X_trv.shape[1],
        lr=best_params["lr"],
        dprob=best_params["dropout"],
        h1=best_params["h1"],
        h2=best_params["h2"]
    )

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=best_params["patience"], restore_best_weights=True
    )

    model.fit(
        X_trv, y_train_full.astype(np.float32),
        validation_split=0.0,  # já usamos todo TREINO+VAL
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=0,
        callbacks=[early]
    )

    # Avaliação no TESTE
    yhat = model.predict(X_te, verbose=0).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    mae  = mean_absolute_error(y_test, yhat)
    r2   = r2_score(y_test, yhat)

    # Salvar artefatos
    outdir = "artefatos_mlp_optuna"
    os.makedirs(outdir, exist_ok=True)

    # Campos de saída (metadados de data/cidade)
    cols_saida = []
    if "Cidade" in df.columns:
        cols_saida.append("Cidade")
    cols_saida += ["Ano", mes_col]

    # Atenção: datas_test e y_test se referem ao recorte de TESTE no df ordenado
    saida = df.loc[df.index[-len(y_test):], cols_saida].copy()
    saida["y_true"] = y_test
    saida["yhat_mlp"] = yhat
    saida["data_ref"] = datas_test.values
    saida.to_csv(f"{outdir}/predicoes_teste.csv", index=False, encoding="utf-8")

    # Salva modelo e scaler
    model.save(f"{outdir}/modelo_mlp.keras")
    np.savez(f"{outdir}/escalador_params.npz",
             features=np.array(features, dtype=object),
             medianas=X_trv_df.median(numeric_only=True).values.astype(np.float32),
             mu=mu.astype(np.float32),
             std=sd.astype(np.float32))

    # Métricas e melhores hiperparâmetros
    with open(f"{outdir}/metricas.json","w",encoding="utf-8") as f:
        json.dump({"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}, f, ensure_ascii=False, indent=2)

    with open(f"{outdir}/best_params.json","w",encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print("\nMétricas no TESTE (mais recente):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print("\nArquivos gerados em ./artefatos_mlp_optuna:")
    print("- predicoes_teste.csv")
    print("- modelo_mlp.keras")
    print("- escalador_params.npz")
    print("- metricas.json")
    print("- best_params.json")

def main():
    # Verbosidade menor do TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    set_seeds(RANDOM_SEED)

    # 1) Dados
    df, target, features, mes_col = load_and_prepare()
    (X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test,
     X_train_full, y_train_full) = temporal_splits(df, features, target)

    # 2) Estudo Optuna
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=(STORAGE is not None)
    )
    objective = objective_factory(X_tr_df, y_tr, X_val_df, y_val, X_test_df, features)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)

    print("\n=== RESULTADOS OPTUNA ===")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  Val RMSE: {study.best_value:.6f}")
    print("  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # 3) Treino final (TREINO+VAL) e avaliação no TESTE
    best_params = study.best_trial.params.copy()
    train_final_and_save(
        best_params, df, features, target, mes_col,
        X_tr_df, y_tr, X_val_df, y_val, X_test_df, y_test, datas_test
    )

    # (Opcional) ranking dos top-5 trials
    print("\nTop-5 trials por RMSE de validação:")
    top5 = sorted(study.trials, key=lambda t: t.value)[:5]
    for t in top5:
        print(f"  Trial #{t.number:>3} | val_RMSE={t.value:.6f} | params={t.params}")

if __name__ == "__main__":
    main()

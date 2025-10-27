# xgb_ok.py
# Requisitos:
#   pip install pandas numpy xgboost joblib openpyxl

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ====================== CONFIG ======================
ARQ = "Modelo Final.xlsx"    # caminho do seu Excel
ABA = None                  # nome da aba (None = primeira)
TEST_SIZE  = 0.20           # fração mais recente para TESTE
VALID_FRAC = 0.15           # fração final do TREINO para VALIDAÇÃO (early stopping)
RANDOM_STATE = 42

N_ESTIMATORS = 2000         # alto o suficiente para o early stopping parar antes
MAX_DEPTH = 6
LEARNING_RATE = 0.05
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
REG_LAMBDA = 2.0
TREE_METHOD = "hist"        # rápido e estável
EARLY_STOPPING_ROUNDS = 50
EVAL_METRIC = "rmse"
# ====================================================

def _ensure_cols(df, cols, alt_map=None, what="feature"):
    """Garante presença das colunas em 'cols'; aceita alternativas em alt_map."""
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

def _to_numeric_and_impute(df_train, df_to_fix, strategy="median"):
    """Converte para numérico e imputa NaN com estatísticas do df_train."""
    # para garantir coerção numérica
    df_train_num = df_train.apply(pd.to_numeric, errors="coerce")
    df_to_fix_num = df_to_fix.apply(pd.to_numeric, errors="coerce")

    if strategy == "median":
        stats = df_train_num.median(numeric_only=True)
    elif strategy == "mean":
        stats = df_train_num.mean(numeric_only=True)
    else:
        raise ValueError("strategy deve ser 'median' ou 'mean'.")

    # aplica imputação
    df_train_num = df_train_num.fillna(stats)
    df_to_fix_num = df_to_fix_num.fillna(stats)

    # se ainda sobrou NaN (ex.: todas NaN em uma coluna), zera
    df_train_num = df_train_num.fillna(0.0)
    df_to_fix_num = df_to_fix_num.fillna(0.0)

    return df_train_num, df_to_fix_num

def main():
    # ---------- 1) Ler dados ----------
    if not os.path.exists(ARQ):
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ}")
    df = pd.read_excel(ARQ, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    df = df.sort_values(['Cidade', 'Ano', 'Mês'])
    agrupado_por_cidade_ano = df.groupby(by=['Cidade'])['Produção de Energia FV'].cumsum()
    df['Produção FV acumulada'] = agrupado_por_cidade_ano


    # Target
    possiveis_targets = ["Produção FV acumulada", "Produção FV acumulada", "Produção", "Producao de Energia"]
    target = next((t for t in possiveis_targets if t in df.columns), None)
    if target is None:
        raise ValueError("Coluna alvo não encontrada (ex.: 'Produção de Energia FV').")

    # Features solicitadas
    features_orig = ["Produção de Energia FV", "Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
    features = _ensure_cols(
        df, features_orig,
        alt_map={"Área": "Area"},
        what="de feature"
    )

    # Colunas de data para split temporal
    mes_col = "Mês" if "Mês" in df.columns else ("Mes" if "Mes" in df.columns else None)
    if "Ano" not in df.columns or mes_col is None:
        raise ValueError("Preciso de 'Ano' e 'Mês/Mes' para split temporal.")

    # ---------- 2) Split temporal ----------
    df["__DATA__"] = pd.to_datetime(dict(
        year=df["Ano"].astype(int),
        month=df[mes_col].astype(int),
        day=1
    ))
    # remove linhas sem target e ordena no tempo
    df = df[~pd.isna(df[target])].sort_values("__DATA__").reset_index(drop=True)

    X_all = df[features].copy()
    y_all = df[target].astype(float).values

    n = len(df)
    n_test = max(1, int(np.floor(TEST_SIZE * n)))
    cut_test = n - n_test

    # Teste = bloco mais recente
    X_train_full = X_all.iloc[:cut_test, :].copy()
    y_train_full = y_all[:cut_test]
    X_test = X_all.iloc[cut_test:, :].copy()
    y_test = y_all[cut_test:]
    datas_test = df["__DATA__"].iloc[cut_test:]

    # ---------- 3) Split interno do treino para validação (early stopping) ----------
    n_tr = len(X_train_full)
    n_val = max(1, int(np.floor(VALID_FRAC * n_tr)))
    cut_val = n_tr - n_val

    X_tr = X_train_full.iloc[:cut_val, :].copy()
    y_tr = y_train_full[:cut_val]
    X_val = X_train_full.iloc[cut_val:, :].copy()
    y_val = y_train_full[cut_val:]

    # ---------- 4) Tratamento simples de NaN ----------
    X_tr, X_val = _to_numeric_and_impute(X_tr, X_val, strategy="median")
    _,    X_test = _to_numeric_and_impute(X_tr, X_test, strategy="median")  # usa stats do X_tr

    # Converte para numpy para não ter "feature_names mismatch"
    X_tr_np  = X_tr.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_te_np  = X_test.values.astype(np.float32)

    # ---------- 5) XGBoost com early stopping ----------
    xgb = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        random_state=RANDOM_STATE,
        tree_method=TREE_METHOD,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric=EVAL_METRIC,
    )

    xgb.fit(
        X_tr_np, y_tr,
        eval_set=[(X_val_np, y_val)],
        verbose=True
    )

    best_it = xgb.best_iteration
    best_score = xgb.best_score
    print(f"best_iteration: {best_it}, best_val_{EVAL_METRIC}: {best_score:.6f}")

    # ---------- 6) Avaliação no TESTE ----------
    if best_it is not None:
        yhat = xgb.predict(X_te_np, iteration_range=(0, best_it + 1))
    else:
        yhat = xgb.predict(X_te_np)

    rmse = mean_squared_error(y_test, yhat)
    mae  = mean_absolute_error(y_test, yhat)
    r2   = r2_score(y_test, yhat)

    print("\nMétricas (teste mais recente):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    # ---------- 7) Salvar artefatos ----------
    os.makedirs("artefatos_xgb", exist_ok=True)

    cols_saida = []
    if "Cidade" in df.columns:
        cols_saida.append("Cidade")
    cols_saida += ["Ano", mes_col]

    saida = df.iloc[cut_test:, :][cols_saida].copy()
    saida["y_true"] = y_test
    saida["yhat_xgb"] = yhat
    saida["data_ref"] = datas_test.values
    saida.to_csv("artefatos_xgb/predicoes_teste.csv", index=False, encoding="utf-8")

    joblib.dump(xgb, "artefatos_xgb/modelo_xgb.pkl")

    with open("artefatos_xgb/metricas.txt", "w", encoding="utf-8") as f:
        f.write(f"best_iteration={best_it}\n")
        f.write(f"best_val_{EVAL_METRIC}={best_score}\n")
        f.write(f"RMSE_test={rmse:.6f}\nMAE_test={mae:.6f}\nR2_test={r2:.6f}\n")

    print("\nArquivos gerados em ./artefatos_xgb:")
    print("- predicoes_teste.csv")
    print("- modelo_xgb.pkl")
    print("- metricas.txt")

if __name__ == "__main__":
    main()

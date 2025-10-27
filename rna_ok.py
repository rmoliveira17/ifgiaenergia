# mlp_ok.py
# Requisitos:
#   pip install pandas numpy tensorflow openpyxl

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====================== CONFIG ======================
ARQ = "Modelo Final.xlsx"    # caminho do seu Excel
ABA = None                  # nome da aba (None = primeira)
TEST_SIZE  = 0.20           # fração mais recente para TESTE
VALID_FRAC = 0.10           # fração final do TREINO para VALIDAÇÃO
RANDOM_SEED = 42

EPOCHS = 300
BATCH_SIZE = 16
LR = 1e-3
DROPOUT = 0.10
H1, H2 = 128, 64           # neurônios das camadas ocultas
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def main():
    # Seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

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

    # Features
    features_orig = ["Produção de Energia FV", "Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
    features = ensure_cols(df, features_orig, alt_map={"Área": "Area"}, what="de feature")

    # Colunas de data para split temporal
    mes_col = "Mês" if "Mês" in df.columns else ("Mes" if "Mes" in df.columns else None)
    if "Ano" not in df.columns or mes_col is None:
        raise ValueError("Preciso de 'Ano' e 'Mês/Mes' para split temporal.")

    # ---------- 2) Split temporal ----------
    df["__DATA__"] = pd.to_datetime(dict(year=df["Ano"].astype(int),
                                         month=df[mes_col].astype(int),
                                         day=1))
    df = df[~pd.isna(df[target])].sort_values("__DATA__").reset_index(drop=True)

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

    # ---------- 3) Split interno do treino (validação temporal) ----------
    n_tr = len(X_train_full)
    n_val = max(1, int(np.floor(VALID_FRAC * n_tr)))
    cut_val = n_tr - n_val

    X_tr_df = X_train_full.iloc[:cut_val, :].copy()
    y_tr    = y_train_full[:cut_val]
    X_val_df= X_train_full.iloc[cut_val:, :].copy()
    y_val   = y_train_full[cut_val:]

    # ---------- 4) Imputação simples + padronização z-score ----------
    X_tr_df, X_val_df, medianas = to_numeric_and_impute(X_tr_df, X_val_df, strategy="median")
    _,      X_test_df, _        = to_numeric_and_impute(X_tr_df, X_test_df, strategy="median")  # usa stats de X_tr_df

    mu, sd = zscore_fit(X_tr_df)
    X_tr = zscore_apply(X_tr_df,  mu, sd)
    X_val = zscore_apply(X_val_df, mu, sd)
    X_te  = zscore_apply(X_test_df, mu, sd)

    y_tr  = y_tr.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_te  = y_test.astype(np.float32)

    # ---------- 5) MLP (EarlyStopping em val_loss) ----------
    model = build_mlp(X_tr.shape[1], lr=LR, dprob=DROPOUT, h1=H1, h2=H2)
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early])

    # ---------- 6) Avaliação no TESTE ----------
    yhat = model.predict(X_te, verbose=0).ravel()
    rmse = mean_squared_error(y_test, yhat)
    mae  = mean_absolute_error(y_test, yhat)
    r2   = r2_score(y_test, yhat)

    print("\nMétricas (teste mais recente):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    # ---------- 7) Salvar artefatos ----------
    os.makedirs("artefatos_mlp", exist_ok=True)

    cols_saida = []
    if "Cidade" in df.columns:
        cols_saida.append("Cidade")
    cols_saida += ["Ano", mes_col]

    saida = df.iloc[cut_test:, :][cols_saida].copy()
    saida["y_true"] = y_test
    saida["yhat_mlp"] = yhat
    saida["data_ref"] = datas_test.values
    saida.to_csv("artefatos_mlp/predicoes_teste.csv", index=False, encoding="utf-8")

    model.save("artefatos_mlp/modelo_mlp.keras")

    # salvar parâmetros do scaler e ordem das features
    np.savez("artefatos_mlp/escalador_params.npz",
             features=np.array(features, dtype=object),
             medianas=medianas.values.astype(np.float32),
             mu=mu.astype(np.float32),
             std=sd.astype(np.float32))

    with open("artefatos_mlp/metricas.json","w",encoding="utf-8") as f:
        json.dump({"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}, f, ensure_ascii=False, indent=2)

    print("\nArquivos gerados em ./artefatos_mlp:")
    print("- predicoes_teste.csv")
    print("- modelo_mlp.keras")
    print("- escalador_params.npz")
    print("- metricas.json")

if __name__ == "__main__":
    main()

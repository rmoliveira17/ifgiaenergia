# -*- coding: utf-8 -*-
# Requisitos:
#   pip install pandas numpy xgboost openpyxl matplotlib

import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====================== CONFIG PADRÃO ======================
ARQ_DEFAULT = "Modelo Final.xlsx"
ABA_DEFAULT = None

ARTEFATOS_DIR = "artefatos_xgb_optuna"
MODELO_JSON   = os.path.join(ARTEFATOS_DIR, "modelo_xgb.json")
BEST_PARAMS   = os.path.join(ARTEFATOS_DIR, "best_params.json")       # opcional
IMPUTER_STATS = os.path.join(ARTEFATOS_DIR, "imputer_stats.json")     # opcional
FEATURES_USED = os.path.join(ARTEFATOS_DIR, "features_used.json")     # opcional

FEATURES_ORIG = ["Produção de Energia FV","Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
ALT_MAP = {"Área": "Area"}  # nome alternativo
# ==========================================================

def _read_any(path: str, sheet_name=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    elif ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    else:
        raise ValueError(f"Extensão não suportada: {ext}")

def _ensure_cols(df: pd.DataFrame, cols, alt_map=None, what="feature"):
    alt_map = alt_map or {}
    out = []
    for c in cols:
        if c in df.columns:
            out.append(c)
        elif c in alt_map and alt_map[c] in df.columns:
            out.append(alt_map[c])
        else:
            raise ValueError(f"Coluna {what} ausente: '{c}' (ou '{alt_map.get(c)}').")
    return out

def _load_imputer_stats():
    if os.path.exists(IMPUTER_STATS):
        with open(IMPUTER_STATS, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return pd.Series(stats, dtype="float64")
    return None

def _load_features_list(df_cols):
    # se existir arquivo de features salvas, usa-o; senão usa lista padrão com ALT_MAP
    if os.path.exists(FEATURES_USED):
        with open(FEATURES_USED, "r", encoding="utf-8") as f:
            feats = json.load(f)
        return _ensure_cols(pd.DataFrame(columns=df_cols), feats, ALT_MAP, what="de feature")
    else:
        return _ensure_cols(pd.DataFrame(columns=df_cols), FEATURES_ORIG, ALT_MAP, what="de feature")

def _coerce_and_impute(df_feats: pd.DataFrame, stats: pd.Series | None):
    X = df_feats.apply(pd.to_numeric, errors="coerce")
    if stats is not None:
        stats_aligned = stats.reindex(X.columns)
        X = X.fillna(stats_aligned).fillna(0.0)
    else:
        med = X.median(numeric_only=True)
        X = X.fillna(med).fillna(0.0)
    return X.astype(np.float32)

def _load_booster(path_model: str) -> xgb.Booster:
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"Modelo não encontrado: {path_model}")
    booster = xgb.Booster()
    booster.load_model(path_model)
    return booster

def _resolve_mes_col(df: pd.DataFrame) -> str:
    if "Mês" in df.columns: return "Mês"
    if "Mes" in df.columns: return "Mes"
    raise ValueError("Coluna de mês não encontrada. Esperado 'Mês' ou 'Mes'.")

def _resolve_target_col(df: pd.DataFrame) -> str:
    # Mesma lógica do treino
    possiveis_targets = ["Produção FV acumulada", "Produção", "Producao de Energia"]
    for t in possiveis_targets:
        if t in df.columns:
            return t
    raise ValueError("Coluna alvo não encontrada (ex.: 'Produção FV acumulada', 'Produção', 'Producao de Energia').")

def prever_comparar(arquivo, aba, cidade, ano, saida_prefixo):
    if not os.path.exists(arquivo):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {arquivo}")

    df = _read_any(arquivo)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(['Cidade', 'Ano', "Mês"])

    # alvo acumulado por cidade (igual ao seu)
    agrupado_por_cidade_ano = df.groupby(by=['Cidade'])['Produção de Energia FV'].cumsum()
    df['Produção FV acumulada'] = agrupado_por_cidade_ano

    mes_col = _resolve_mes_col(df)
    target  = _resolve_target_col(df)

    _ensure_cols(df, ["Cidade", "Ano"], what="de identificação")
    df["Cidade"] = df["Cidade"].astype(str)

    filtro = (df["Cidade"] == str(cidade)) & (df["Ano"].astype(int) == int(ano))
    dfx = df.loc[filtro].copy()
    if dfx.empty:
        raise ValueError(f"Nenhuma linha encontrada para cidade='{cidade}' e ano={ano}.")

    dfx = dfx.sort_values(by=[mes_col]).reset_index(drop=True)

    # features
    feats = _load_features_list(df.columns)
    X_num = _coerce_and_impute(dfx[feats].copy(), _load_imputer_stats())

    # booster
    booster = _load_booster(MODELO_JSON)

    # previsão
    dmatrix = xgb.DMatrix(X_num.values)
    yhat = booster.predict(dmatrix)

    # comparação
    comp = dfx[["Cidade", "Ano", mes_col]].copy()
    comp["y_true"] = pd.to_numeric(dfx[target], errors="coerce")
    comp["yhat_xgb"] = yhat
    comp["erro"] = comp["y_true"] - comp["yhat_xgb"]

    # métricas (mín. 2 pontos válidos p/ R²)
    comp_valid = comp.dropna(subset=["y_true"]).copy()
    if len(comp_valid) >= 2:
        rmse = float(np.sqrt(mean_squared_error(comp_valid["y_true"], comp_valid["yhat_xgb"])))
        mae  = float(mean_absolute_error(comp_valid["y_true"], comp_valid["yhat_xgb"]))
        r2   = float(r2_score(comp_valid["y_true"], comp_valid["yhat_xgb"]))
    elif len(comp_valid) == 1:
        rmse = float(np.sqrt(mean_squared_error(comp_valid["y_true"], comp_valid["yhat_xgb"])))
        mae  = float(mean_absolute_error(comp_valid["y_true"], comp_valid["yhat_xgb"]))
        r2   = np.nan
    else:
        rmse = mae = r2 = np.nan

    # salva Excel
    os.makedirs("saidas_previsao_xgb", exist_ok=True)
    base = f"{saida_prefixo}_{cidade}_{ano}".replace(" ", "_")
    arq_tab = os.path.join("saidas_previsao_xgb", f"comparacao_{base}.xlsx")
    comp.to_excel(arq_tab, index=False, engine="openpyxl")

    # plot robusto: plota o que houver
    fig, ax = plt.subplots(figsize=(10, 5))
    meses = pd.to_numeric(comp[mes_col], errors="coerce").values

    if comp["y_true"].notna().any():
        ax.plot(meses, comp["y_true"].values, marker="o", label="Real")
    ax.plot(meses, comp["yhat_xgb"].values, marker="s", label="Previsto")

    ax.set_xlabel("Mês")
    ax.set_ylabel(target)
    ax.set_title(f"{cidade} - {ano}: {target} (Real vs Previsto)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    arq_fig = os.path.join("saidas_previsao_xgb", f"fig_previsao_{base}.png")
    fig.savefig(arq_fig, dpi=150)
    plt.close(fig)

    print("\n===== RESUMO =====")
    print(f"Cidade/Ano: {cidade} / {ano}")
    print(f"Linhas avaliadas: {len(comp)}")
    if not np.isnan(rmse):
        r2_txt = f"{r2:.4f}" if not np.isnan(r2) else "NA"
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2_txt}")
    else:
        print("Sem valores reais suficientes para calcular métricas (y_true ausente).")
    print(f"Planilha: {arq_tab}")
    print(f"Figura  : {arq_fig}")

    if os.path.exists(BEST_PARAMS):
        with open(BEST_PARAMS, "r", encoding="utf-8") as f:
            bp = json.load(f)
        print("\nHiperparâmetros do melhor trial (Optuna):")
        print(json.dumps(bp, ensure_ascii=False, indent=2))

def main():
    p = argparse.ArgumentParser(description="Prever com Booster salvo e comparar/plotar por cidade e ano.")
    p.add_argument("--cidade", "-c", required=True, help="Nome exato da cidade (igual no dataset).")
    p.add_argument("--ano", "-a", required=True, type=int, help="Ano (ex.: 2024).")
    p.add_argument("--arquivo", "-i", default=ARQ_DEFAULT, help="Caminho do Excel/CSV com os dados (padrão: Modelo Final.xlsx).")
    p.add_argument("--aba", "-s", default=ABA_DEFAULT, help="Nome da planilha (Excel).")
    p.add_argument("--prefixo", "-p", default="xgb", help="Prefixo para arquivos de saída.")
    args = p.parse_args()

    prever_comparar(args.arquivo, args.aba, args.cidade, args.ano, args.prefixo)

if __name__ == "__main__":
    main()

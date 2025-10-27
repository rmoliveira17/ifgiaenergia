# -*- coding: utf-8 -*-
# Requisitos:
#   pip install pandas numpy tensorflow openpyxl matplotlib

import os
import json
import argparse
import pdb

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras

# ====================== CONFIG PADRÃO ======================
ARQ_DEFAULT = "Modelo Final.xlsx"
ABA_DEFAULT = None

ARTEFATOS_DIR = "artefatos_mlp"
MODEL_PATH    = os.path.join(ARTEFATOS_DIR, "modelo_mlp.keras")
IMPUTER_STATS = os.path.join(ARTEFATOS_DIR, "imputer_stats.json")   # opcional (medianas)
SCALER_STATS  = os.path.join(ARTEFATOS_DIR, "scaler_stats.json")    # opcional (mean/std)
FEATURES_USED = os.path.join(ARTEFATOS_DIR, "features_used.json")   # opcional

FEATURES_ORIG = ["Produção de Energia FV","Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
ALT_MAP = {"Área": "Area"}  # alias
# ==========================================================

logger = logging.getLogger("mlp_predict_compare")

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

def _resolve_mes_col(df: pd.DataFrame) -> str:
    if "Mês" in df.columns: return "Mês"
    if "Mes" in df.columns: return "Mes"
    raise ValueError("Coluna de mês não encontrada. Esperado 'Mês' ou 'Mes'.")

def _resolve_target_col(df: pd.DataFrame) -> str:
    possiveis_targets = ["Produção FV acumulada", "Produção", "Producao de Energia"]
    for t in possiveis_targets:
        if t in df.columns:
            return t
    raise ValueError("Coluna alvo não encontrada (ex.: 'Produção FV acumulada', 'Produção', 'Producao de Energia').")

def _load_imputer_stats():
    if os.path.exists(IMPUTER_STATS):
        with open(IMPUTER_STATS, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return pd.Series(stats, dtype="float64")
    return None

def _load_scaler_stats():
    if os.path.exists(SCALER_STATS):
        with open(SCALER_STATS, "r", encoding="utf-8") as f:
            d = json.load(f)
        mean = pd.Series(d.get("mean", {}), dtype="float64")
        std  = pd.Series(d.get("std",  {}), dtype="float64")
        return mean, std
    return None, None

def _load_features_saved(df_cols):
    if os.path.exists(FEATURES_USED):
        with open(FEATURES_USED, "r", encoding="utf-8") as f:
            feats = json.load(f)
        # valida com ALT_MAP
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

def _scale_zscore(X: pd.DataFrame, mean_stats: pd.Series | None, std_stats: pd.Series | None):
    if mean_stats is not None and std_stats is not None and len(mean_stats) and len(std_stats):
        mu  = mean_stats.reindex(X.columns).astype(float)
        std = std_stats.reindex(X.columns).astype(float).replace(0.0, 1.0)
        Z = (X - mu) / std
        return Z.astype(np.float32)
    else:
        mu  = X.mean(numeric_only=True)
        std = X.std(numeric_only=True).replace(0.0, 1.0)
        Z = (X - mu) / std
        return Z.astype(np.float32)

def _load_model(path_model: str):
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"Modelo não encontrado: {path_model}")
    model = keras.models.load_model(path_model)
    return model

def _reconcile_features_with_model(feats, df, model, strict=True):
    """Garante que o número/ordem de feats bata com o que o modelo espera.
       Tenta usar features_used.json; se não houver, remove 'suspeitos' até bater."""
    n_expected = int(model.input_shape[-1])
    logger.info("Modelo espera %d features.", n_expected)

    if len(feats) == n_expected:
        logger.info("Features já compatíveis (%d).", len(feats))
        return feats

    logger.warning("Qtd de features não bate (tem %d, modelo espera %d). Tentando reconciliar...", len(feats), n_expected)

    # 1) Se existir features_used.json, confie nele
    if os.path.exists(FEATURES_USED):
        with open(FEATURES_USED, "r", encoding="utf-8") as f:
            feats_saved = json.load(f)
        # valida presença
        feats_saved = _ensure_cols(pd.DataFrame(columns=df.columns), feats_saved, ALT_MAP, what="de feature")
        if len(feats_saved) != n_expected:
            raise ValueError(
                f"'features_used.json' tem {len(feats_saved)} colunas, mas o modelo espera {n_expected}. "
                f"Ajuste o treino ou salve o arquivo correto."
            )
        logger.info("Usando features de 'features_used.json'.")
        return feats_saved

    # 2) Tentativa automática: remover candidatos prováveis
    candidates = feats.copy()

    # alta probabilidade de NÃO ser feature de entrada no MLP: alvo(s) e, às vezes, Área
    drop_priority = [
        "Produção FV acumulada",    # alvo acumulado criado no pré-processo
    ]

    for col in drop_priority:
        if len(candidates) <= n_expected:
            break
        if col in candidates:
            candidates.remove(col)
            logger.info("Removendo coluna '%s' para reconciliar.", col)

    # Se ainda sobrar mais do que o esperado, remova do fim (últimos) até bater
    while len(candidates) > n_expected:
        removed = candidates.pop()  # remove a última
        logger.info("Removendo excedente '%s' para igualar dimensão.", removed)

    # Se ficar faltando, aborta com instrução clara
    if len(candidates) != n_expected:
        raise ValueError(
            f"Ainda faltam/sobram colunas para casar com o modelo. "
            f"Temos {len(candidates)} vs {n_expected}. "
            f"Sugestão: no treino do MLP, salve 'features_used.json' com a lista exata, "
            f"e use-o aqui."
        )

    logger.info("Features reconciliadas: %s", candidates)
    return candidates

def prever_comparar(arquivo, aba, cidade, ano, saida_prefixo, debug=False):
    if not os.path.exists(arquivo):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {arquivo}")

    df = _read_any(arquivo)
    df.columns = [c.strip() for c in df.columns]

    # Ordenação e criação do acumulado (se necessário, garantindo consistência com seu pipeline)
    mes_col_guess = "Mês" if "Mês" in df.columns else ("Mes" if "Mes" in df.columns else None)
    if not {"Cidade", "Ano"}.issubset(df.columns) or mes_col_guess is None:
        raise ValueError("Preciso de 'Cidade', 'Ano' e 'Mês/Mes' no arquivo.")

    df = df.sort_values(['Cidade', 'Ano', mes_col_guess])
    if "Produção FV acumulada" not in df.columns and "Produção de Energia FV" in df.columns:
        df["Produção FV acumulada"] = df.groupby('Cidade')["Produção de Energia FV"].cumsum()

    mes_col = _resolve_mes_col(df)
    target  = _resolve_target_col(df)

    _ensure_cols(df, ["Cidade", "Ano"], what="de identificação")
    df["Cidade"] = df["Cidade"].astype(str)

    # Filtro Cidade/Ano
    filtro = (df["Cidade"] == str(cidade)) & (df["Ano"].astype(int) == int(ano))
    dfx = df.loc[filtro].copy()
    if dfx.empty:
        raise ValueError(f"Nenhuma linha encontrada para cidade='{cidade}' e ano={ano}.")

    dfx = dfx.sort_values(by=[mes_col]).reset_index(drop=True)

    # Carrega modelo e descobre dimensão esperada
    model = _load_model(MODEL_PATH)

    # Carrega lista base de features e reconcilia com o modelo
    feats_base = _load_features_saved(df.columns)
    feats = _reconcile_features_with_model(feats_base, df, model)
    pdb.set_trace()
    if debug:
        logger.debug("Features finais usadas (%d): %s", len(feats), feats)

    # Imputação + escala
    X_num = _coerce_and_impute(dfx[feats].copy(), _load_imputer_stats())
    mu, sd = _load_scaler_stats()
    X_std = _scale_zscore(X_num, mu, sd)

    # Predição
    yhat = model.predict(X_std, verbose=0).reshape(-1)

    # Comparação
    comp = dfx[["Cidade", "Ano", mes_col]].copy()
    comp["y_true"] = pd.to_numeric(dfx[target], errors="coerce")
    comp["yhat_mlp"] = yhat
    comp["erro"] = comp["y_true"] - comp["yhat_mlp"]

    # Métricas
    comp_valid = comp.dropna(subset=["y_true"]).copy()
    if len(comp_valid) >= 2:
        rmse = float(np.sqrt(mean_squared_error(comp_valid["y_true"], comp_valid["yhat_mlp"])))
        mae  = float(mean_absolute_error(comp_valid["y_true"], comp_valid["yhat_mlp"]))
        r2   = float(r2_score(comp_valid["y_true"], comp_valid["yhat_mlp"]))
    elif len(comp_valid) == 1:
        rmse = float(np.sqrt(mean_squared_error(comp_valid["y_true"], comp_valid["yhat_mlp"])))
        mae  = float(mean_absolute_error(comp_valid["y_true"], comp_valid["yhat_mlp"]))
        r2   = np.nan
    else:
        rmse = mae = r2 = np.nan

    # Saídas
    outdir = "saidas_previsao_mlp"
    os.makedirs(outdir, exist_ok=True)
    base = f"{saida_prefixo}_{cidade}_{ano}".replace(" ", "_")

    arq_tab = os.path.join(outdir, f"comparacao_{base}.xlsx")
    comp.to_excel(arq_tab, index=False, engine="openpyxl")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    meses = pd.to_numeric(comp[mes_col], errors="coerce").values
    if comp["y_true"].notna().any():
        ax.plot(meses, comp["y_true"].values, marker="o", label="Real")
    ax.plot(meses, comp["yhat_mlp"].values, marker="s", label="Previsto")
    ax.set_xlabel("Mês")
    ax.set_ylabel(target)
    ax.set_title(f"{cidade} - {ano}: {target} (Real vs Previsto) [MLP]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    arq_fig = os.path.join(outdir, f"fig_previsao_{base}.png")
    fig.savefig(arq_fig, dpi=150)
    plt.close(fig)

    print("\n===== RESUMO (MLP) =====")
    print(f"Cidade/Ano: {cidade} / {ano}")
    print(f"Linhas avaliadas: {len(comp)}")
    if not np.isnan(rmse):
        r2_txt = f"{r2:.4f}" if not np.isnan(r2) else "NA"
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2_txt}")
    else:
        print("Sem valores reais suficientes para calcular métricas (y_true ausente).")
    print(f"Planilha: {arq_tab}")
    print(f"Figura  : {arq_fig}")

def main():
    p = argparse.ArgumentParser(description="Prever com MLP salvo e comparar/plotar por cidade e ano.")
    p.add_argument("--cidade", "-c", required=True, help="Nome exato da cidade (igual no dataset).")
    p.add_argument("--ano", "-a", required=True, type=int, help="Ano (ex.: 2024).")
    p.add_argument("--arquivo", "-i", default=ARQ_DEFAULT, help="Caminho do Excel/CSV com os dados (padrão: Modelo Final.xlsx).")
    p.add_argument("--aba", "-s", default=ABA_DEFAULT, help="Nome da planilha (Excel).")
    p.add_argument("--prefixo", "-p", default="mlp", help="Prefixo para arquivos de saída.")
    args = p.parse_args()

    prever_comparar(args.arquivo, args.aba, args.cidade, args.ano, args.prefixo)

if __name__ == "__main__":
    main()

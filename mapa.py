# -*- coding: utf-8 -*-
# Requisitos:
#   pip install pandas numpy xgboost geopandas shapely matplotlib openpyxl geobr

import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

# ====================== CONFIG PADRÃO ======================
ARQ_DEFAULT      = "Modelo Final.xlsx"          # seu dataset
ABA_DEFAULT      = None
ARTEFATOS_DIR    = "artefatos_xgb_optuna"
MODELO_JSON      = os.path.join(ARTEFATOS_DIR, "modelo_xgb.json")
IMPUTER_STATS    = os.path.join(ARTEFATOS_DIR, "imputer_stats.json")   # opcional
FEATURES_USED    = os.path.join(ARTEFATOS_DIR, "features_used.json")   # opcional
OUT_DIR          = "saidas_mapa_xgb"
GOIAS_SHAPE      = None  # se quiser apontar para um arquivo local (shp/geojson). Caso None, tenta usar geobr.

FEATURES_ORIG = ["Produção de Energia FV","Consumo","Lat","Long","Temperatura","DNI","DHI","Umidade","Área"]
ALT_MAP = {"Área": "Area"}  # alias
ID_COLS = ["Cidade", "Ano", "Mês", "Mes", "Lat", "Long"]  # usadas/úteis
# ==========================================================

def _read_any(path: str) -> pd.DataFrame:
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
    # Usado apenas para comparação (quando houver), e para inferir tipo de agregação
    possiveis_targets = ["Produção FV acumulada", "Produção", "Producao de Energia"]
    for t in possiveis_targets:
        if t in df.columns:
            return t
    # Se não houver nenhum alvo no arquivo de inferência, seguimos só com previsão
    return None

def _load_imputer_stats():
    if os.path.exists(IMPUTER_STATS):
        with open(IMPUTER_STATS, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return pd.Series(stats, dtype="float64")
    return None

def _load_features_list(df_cols):
    if os.path.exists(FEATURES_USED):
        with open(FEATURES_USED, "r", encoding="utf-8") as f:
            feats = json.load(f)
        return _ensure_cols(pd.DataFrame(columns=df_cols), feats, ALT_MAP, what="de feature")
    else:
        return _ensure_cols(pd.DataFrame(columns=df_cols), FEATURES_ORIG, ALT_MAP, what="de feature")

def _to_numeric_and_impute(X_df: pd.DataFrame, stats: pd.Series | None):
    X = X_df.apply(pd.to_numeric, errors="coerce")
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

def _load_goias_geometry():
    # 1) Se o usuário forneceu um shape local, usa
    if GOIAS_SHAPE and os.path.exists(GOIAS_SHAPE):
        gdf = gpd.read_file(GOIAS_SHAPE)
        # tenta garantir CRS
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    # 2) Tenta via geobr (código do estado de GO = 52)
    try:
        import geobr
        gdf = geobr.read_state(code_state=52, year=2020)  # Goiás
        if gdf.crs is None:
            gdf.set_crs(epsg=4674, inplace=True)  # SIRGAS 2000 (geobr costuma vir com 4674)
        # Converte para WGS84 p/ casar com Lat/Long
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        raise RuntimeError(
            "Não consegui carregar o polígono de Goiás. "
            "Instale o pacote 'geobr' (pip install geobr) OU defina GOIAS_SHAPE para um arquivo local."
        ) from e

def _infer_aggregation_mode(df: pd.DataFrame, target_col: str | None):
    """
    Se target mensal existir no arquivo com nome típico, assume 'mensal' (somar).
    Se target acumulado existir ('Produção FV acumulada'), assume 'acumulado' (pegar último do ano).
    Se não existir target, tenta inferir por presença da coluna acumulada no DataFrame original.
    """
    if target_col == "Produção FV acumulada":
        return "acumulado"
    if target_col in ("Produção", "Producao de Energia", "Produção de Energia FV"):
        return "mensal"
    # fallback: se a coluna de acumulado existir no DF, é provável que o treino tenha sido acumulado
    if "Produção FV acumulada" in df.columns:
        return "acumulado"
    return "mensal"

def rodar(arquivo, aba, ano, out_prefix):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ======== 1) Lê dados ========
    df = _read_any(arquivo)
    df.columns = [c.strip() for c in df.columns]

    # Campos básicos
    mes_col = _resolve_mes_col(df)
    target  = _resolve_target_col(df)

    # Ordena: garante ordem temporal por cidade
    if not {"Cidade", "Ano"}.issubset(df.columns):
        raise ValueError("Preciso de 'Cidade' e 'Ano' no arquivo.")
    df["Cidade"] = df["Cidade"].astype(str)
    df = df.sort_values(["Cidade", "Ano", mes_col]).reset_index(drop=True)

    # Se tiver a coluna mensal e NÃO tiver acumulado, cria acumulado (consistência com o pipeline anterior)
    if "Produção de Energia FV" in df.columns and "Produção FV acumulada" not in df.columns:
        df["Produção FV acumulada"] = df.groupby("Cidade")["Produção de Energia FV"].cumsum()

    # ======== 2) Features e previsões linha a linha ========
    feats = _load_features_list(df.columns)
    X_all = _to_numeric_and_impute(df[feats].copy(), _load_imputer_stats())

    booster = _load_booster(MODELO_JSON)
    dmat = xgb.DMatrix(X_all.values)
    df["yhat_xgb"] = booster.predict(dmat)

    # ======== 3) Agregação anual por cidade ========
    # Filtra o ano desejado:
    df_ano = df.loc[df["Ano"].astype(int) == int(ano)].copy()
    if df_ano.empty:
        raise ValueError(f"Não há linhas para o ano {ano}.")

    # Decide se soma (mensal) ou pega o último (acumulado)
    modo = _infer_aggregation_mode(df, target)
    if modo == "mensal":
        agg = df_ano.groupby("Cidade", as_index=False).agg({
            "yhat_xgb": "sum",
            "Lat": "first",
            "Long": "first"
        }).rename(columns={"yhat_xgb": "yhat_anual"})
    else:  # acumulado
        # pega o último mês disponível por cidade no ano
        idx_ult = df_ano.groupby("Cidade")[mes_col].transform("idxmax")
        ultimos = df_ano.loc[idx_ult.unique(), ["Cidade", "yhat_xgb", "Lat", "Long"]].copy()
        ultimos = ultimos.rename(columns={"yhat_xgb": "yhat_anual"})
        agg = ultimos.reset_index(drop=True)

    # Remove coords faltantes
    agg = agg.dropna(subset=["Lat", "Long"])

    # ======== 4) Geo (pontos e polígono de GO) ========
    go_gdf = _load_goias_geometry()

    # GeoDataFrame de pontos
    gpoints = gpd.GeoDataFrame(
        agg,
        geometry=[Point(lon, lat) for lat, lon in zip(agg["Lat"].astype(float), agg["Long"].astype(float))],
        crs="EPSG:4326"
    )

    # Mantém apenas pontos dentro de Goiás (caso seu arquivo tenha cidades de outros estados)
    gpoints = gpd.sjoin(gpoints, go_gdf[["geometry"]], how="inner", predicate="within")
    gpoints = gpoints.drop(columns=[c for c in gpoints.columns if c.startswith("index_")], errors="ignore")

    # ======== 5) Plot ========
    fig, ax = plt.subplots(figsize=(8, 8))
    go_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1)

    # bolhas: tamanho proporcional e cor contínua
    # normaliza tamanhos (para não ficar gigante)
    v = gpoints["yhat_anual"].astype(float)
    if len(v) > 0 and v.max() > 0:
        sizes = 50 + 300 * (v / v.max())  # min 50, até 350
    else:
        sizes = 80

    sc = gpoints.plot(
        ax=ax,
        markersize=sizes,
        column="yhat_anual",
        cmap="viridis",
        legend=True,
        alpha=0.85
    )

    ax.set_title(f"GO - Previsão anual acumulada (XGBoost) - {ano}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    map_path = os.path.join(OUT_DIR, f"mapa_goias_xgb_{ano}_{out_prefix}.png")
    plt.savefig(map_path, dpi=180)
    plt.close(fig)

    # ======== 6) Exporta CSV ========
    csv_path = os.path.join(OUT_DIR, f"pred_anual_cidade_go_{ano}_{out_prefix}.csv")
    gpoints.drop(columns="geometry").to_csv(csv_path, index=False, encoding="utf-8")

    print("\n==== OK ====")
    print(f"Modo de agregação: {'soma (mensal)' if modo=='mensal' else 'último do ano (acumulado)'}")
    print(f"Linhas (cidades) no mapa: {len(gpoints)}")
    print(f"CSV:  {csv_path}")
    print(f"Mapa: {map_path}")

def main():
    ap = argparse.ArgumentParser(description="Prever XGBoost para todas as cidades e plotar mapa de GO com acumulado anual.")
    ap.add_argument("--arquivo", "-i", default=ARQ_DEFAULT, help="Caminho do Excel/CSV com os dados.")
    ap.add_argument("--aba", "-s", default=ABA_DEFAULT, help="Aba do Excel (se aplicável).")
    ap.add_argument("--ano", "-a", required=True, type=int, help="Ano a agregar (ex.: 2024).")
    ap.add_argument("--prefixo", "-p", default="xgb", help="Prefixo para arquivos de saída.")
    args = ap.parse_args()

    rodar(args.arquivo, args.aba, args.ano, args.prefixo)

if __name__ == "__main__":
    main()

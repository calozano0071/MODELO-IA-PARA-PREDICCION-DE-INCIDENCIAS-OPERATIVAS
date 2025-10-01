import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


def cargar_datos(path):
    """
    Carga los datos desde un archivo Excel y normaliza nombres de columnas.
    """
    df = pd.read_excel(path)

    # Normalizamos nombres de columnas
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )

    # Renombramos "tipo_de_falla" -> "tipo_falla" para consistencia
    if "tipo_de_falla" in df.columns:
        df = df.rename(columns={"tipo_de_falla": "tipo_falla"})

    # Aseguramos que la columna fecha exista y sea datetime
    if "hora_inicio" in df.columns:
        df["hora_inicio"] = pd.to_datetime(df["hora_inicio"], errors="coerce")

    # Ordenamos por host y hora
    if "host" in df.columns and "hora_inicio" in df.columns:
        df = df.sort_values(["host", "hora_inicio"]).reset_index(drop=True)

    return df


def encode_labels(df, out_dir="models"):
    """Codifica las etiquetas categóricas host y tipo_falla."""
    os.makedirs(out_dir, exist_ok=True)

    le_host = LabelEncoder()
    df["host_id"] = le_host.fit_transform(df["host"])

    le_tipo = LabelEncoder()
    df["tipo_id"] = le_tipo.fit_transform(df["tipo_falla"])

    joblib.dump(le_host, os.path.join(out_dir, "le_host.pkl"))
    joblib.dump(le_tipo, os.path.join(out_dir, "le_tipo.pkl"))

    return df, le_host, le_tipo


def _make_windows(df, le_host, le_tipo, ventana=30):
    """
    Función auxiliar: crea ventanas para un horizonte dado.
    """
    # Normalizar nombres
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )

    if "hora_inicio" not in df.columns:
        raise ValueError("El archivo debe contener la columna 'hora_inicio'")

    df["hora_inicio"] = pd.to_datetime(df["hora_inicio"], errors="coerce")
    df = df.sort_values("hora_inicio").reset_index(drop=True)

    # Codificar host y tipo
    df["host_id"] = le_host.transform(df["host"])
    df["tipo_id"] = le_tipo.transform(df["tipo_falla"])

    # Crear series de conteo diario
    df["dia"] = df["hora_inicio"].dt.normalize()
    conteos = df.groupby(["dia", "host_id", "tipo_id"]).size().reset_index(name="conteo")

    # Expandir series por host/tipo
    dias = pd.date_range(conteos["dia"].min(), conteos["dia"].max(), freq="D")
    combos = conteos[["host_id", "tipo_id"]].drop_duplicates()

    registros = []
    for _, row in combos.iterrows():
        h, t = row["host_id"], row["tipo_id"]
        serie = pd.DataFrame({"dia": dias})
        serie["host_id"] = h
        serie["tipo_id"] = t
        serie["dia"] = pd.to_datetime(serie["dia"]).dt.normalize()
        conteos["dia"] = pd.to_datetime(conteos["dia"]).dt.normalize()
        serie = serie.merge(conteos, on=["dia", "host_id", "tipo_id"], how="left").fillna(0)
        registros.append(serie)

    full = pd.concat(registros, ignore_index=True)

    # --- Crear ventanas ---
    X_seq, X_static, X_host, X_tipo, metas = [], [], [], [], []
    for (h, t), grupo in full.groupby(["host_id", "tipo_id"]):
        valores = grupo["conteo"].values
        fechas = grupo["dia"].values

        for i in range(len(valores) - ventana):
            seq = valores[i:i+ventana]
            X_seq.append(seq.reshape(-1, 1))
            X_static.append([np.mean(seq), np.std(seq), np.min(seq), np.max(seq), ventana, t])
            X_host.append([h])
            X_tipo.append([t])
            metas.append({
                "host": le_host.inverse_transform([h])[0],
                "tipo": le_tipo.inverse_transform([t])[0],
                "fecha": fechas[i+ventana]
            })

    return (
        np.array(X_seq),
        np.array(X_static),
        np.array(X_host),
        np.array(X_tipo),
        pd.DataFrame(metas)
    )


def preparar_datos_prediccion(df, le_host, le_tipo, ventanas=[1, 7, 30]):
    """
    Prepara datos para generar predicciones en varios horizontes (día, semana, mes).
    Devuelve un diccionario {horizonte: (X_seq, X_static, X_host, X_tipo, meta_df)}.
    """
    resultados = {}
    for v in ventanas:
        resultados[v] = _make_windows(df, le_host, le_tipo, ventana=v)
    return resultados

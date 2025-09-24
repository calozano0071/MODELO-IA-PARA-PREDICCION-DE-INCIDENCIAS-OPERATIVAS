import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

def cargar_datos(path):
    """
    Carga los datos desde un archivo Excel y normaliza nombres de columnas.
    """
    df = pd.read_excel(path)

    # Normalizamos nombres de columnas (ej: "Hora Inicio" -> "hora_inicio")
    df.columns = (
        df.columns.str.strip()     # quitamos espacios
                  .str.lower()     # pasamos a minúsculas
                  .str.replace(" ", "_")  # reemplazamos espacios por "_"
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
    df["host_id"] = le_host.fit_transform(df["host"])   # 👈 ahora se llama host_id

    le_tipo = LabelEncoder()
    df["tipo_id"] = le_tipo.fit_transform(df["tipo_falla"])   # 👈 ahora se llama tipo_id

    joblib.dump(le_host, os.path.join(out_dir, "le_host.pkl"))
    joblib.dump(le_tipo, os.path.join(out_dir, "le_tipo.pkl"))

    return df, le_host, le_tipo

import numpy as np
import pandas as pd

def preparar_datos_prediccion(df, le_host, le_tipo, ventana=180):
    """
    Prepara datos de un archivo nuevo para generar predicciones.
    Devuelve X_seq, X_static, X_host y un DataFrame meta (host, tipo, fecha).
    """
    # Aseguramos formato de fecha
    df["Hora Inicio"] = pd.to_datetime(df["Hora Inicio"])
    df = df.sort_values("Hora Inicio").reset_index(drop=True)

    # Codificar host y tipo con los encoders entrenados
    df["host_enc"] = le_host.transform(df["Host"])
    df["tipo_enc"] = le_tipo.transform(df["Tipo de falla"])

    # Crear series de conteo diario (como en entrenamiento)
    df["dia"] = df["Hora Inicio"].dt.date
    conteos = df.groupby(["dia", "host_enc", "tipo_enc"]).size().reset_index(name="conteo")

    # Expandir a serie temporal completa
    dias = pd.date_range(conteos["dia"].min(), conteos["dia"].max(), freq="D")
    combos = conteos[["host_enc", "tipo_enc"]].drop_duplicates()

    registros = []
    for _, row in combos.iterrows():
        h, t = row["host_enc"], row["tipo_enc"]
        serie = pd.DataFrame({"dia": dias})
        serie["host_enc"] = h
        serie["tipo_enc"] = t
        serie = serie.merge(conteos, on=["dia", "host_enc", "tipo_enc"], how="left").fillna(0)
        registros.append(serie)

    full = pd.concat(registros, ignore_index=True)

    # --- Crear ventanas ---
    X_seq, X_static, X_host, metas = [], [], [], []
    for (h, t), grupo in full.groupby(["host_enc", "tipo_enc"]):
        valores = grupo["conteo"].values
        fechas = grupo["dia"].values

        for i in range(len(valores) - ventana):
            seq = valores[i:i+ventana]
            X_seq.append(seq.reshape(-1, 1))
            X_static.append([np.mean(seq), np.std(seq), np.min(seq), np.max(seq), ventana, t])
            X_host.append([h])
            metas.append({
                "host": le_host.inverse_transform([h])[0],
                "tipo": le_tipo.inverse_transform([t])[0],
                "fecha": fechas[i+ventana]
            })

    X_seq = np.array(X_seq)
    X_static = np.array(X_static)
    X_host = np.array(X_host)
    meta_df = pd.DataFrame(metas)

    return X_seq, X_static, X_host, meta_df


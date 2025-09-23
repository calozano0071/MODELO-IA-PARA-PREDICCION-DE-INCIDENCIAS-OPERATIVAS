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

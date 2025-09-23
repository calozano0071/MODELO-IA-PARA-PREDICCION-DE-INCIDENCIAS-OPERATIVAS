import pandas as pd

def cargar_datos(path):
    """
    Carga los datos desde un archivo Excel.
    """
    # Leemos el Excel
    df = pd.read_excel(path)

    # Aseguramos que la columna fecha sea de tipo datetime
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Ordenamos por host y fecha
    if "host" in df.columns and "fecha" in df.columns:
        df = df.sort_values(["host", "fecha"]).reset_index(drop=True)

    return df

import numpy as np
import pandas as pd


def make_daily_aggregates(df):
    """
    Devuelve un DataFrame agregado por día, host y tipo_falla con conteo de fallas.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["hora_inicio"]).dt.floor("D")
    agg = df.groupby(
        ["host", "tipo_falla", "host_id", "tipo_id", "date"]
    ).size().reset_index(name="n_fallas")
    return agg


def build_time_series_for_combo(agg_df, host, tipo, window=180,
                                full_range_min=None, full_range_max=None):
    """
    Construye secuencia temporal con features estáticas para un host-tipo.
    Retorna listas: seqs (window,1), statics, ys, dates
    """
    g = agg_df[(agg_df["host"] == host) & (agg_df["tipo_falla"] == tipo)].sort_values("date")
    if g.empty:
        return [], [], [], []

    # Rango completo de fechas
    if full_range_min is None:
        full_range_min = g["date"].min()
    if full_range_max is None:
        full_range_max = g["date"].max()

    dr = pd.date_range(full_range_min, full_range_max, freq="D")
    s = pd.Series(0, index=dr)
    counts = g.set_index("date")["n_fallas"]
    s.loc[counts.index] = counts.values

    df = pd.DataFrame({"date": dr, "n_fallas": s.values})
    df["y"] = (df["n_fallas"] > 0).astype(int)

    # Features estáticas rolling
    df["sum_7"] = df["y"].rolling(7, min_periods=1).sum().values
    df["sum_30"] = df["y"].rolling(30, min_periods=1).sum().values
    df["sum_90"] = df["y"].rolling(90, min_periods=1).sum().values
    df["sum_180"] = df["y"].rolling(180, min_periods=1).sum().values
    df["dow"] = df["date"].dt.weekday
    df["day_of_month"] = df["date"].dt.day

    # Construcción de ventanas
    seqs, statics, ys, dates = [], [], [], []
    for i in range(window, len(df) - 1):
        seq = df["y"].values[i - window:i].reshape(window, 1).astype(np.float32)
        static = df.loc[i, ["sum_7", "sum_30", "sum_90", "sum_180",
                            "dow", "day_of_month"]].values.astype(np.float32)
        y = int(df.loc[i + 1, "y"])
        target_date = df.loc[i + 1, "date"]

        seqs.append(seq)
        statics.append(static)
        ys.append(y)
        dates.append(target_date)

    return seqs, statics, ys, dates


def preparar_series(df, window=180):
    """
    Genera ventanas para TODOS los host-tipo.
    Devuelve arrays listos para el entrenamiento.
    """
    agg = make_daily_aggregates(df)

    X_seq, X_static, y_all, dates_all, meta = [], [], [], [], []
    for (h, t), _ in agg.groupby(["host", "tipo_falla"]):
        seqs, statics, ys, dates = build_time_series_for_combo(agg, h, t, window)
        X_seq.extend(seqs)
        X_static.extend(statics)
        y_all.extend(ys)
        dates_all.extend(dates)
        meta.extend([(h, t)] * len(ys))

    return (
        np.array(X_seq),
        np.array(X_static),
        np.array(y_all),
        np.array(dates_all),
        pd.DataFrame(meta, columns=["host", "tipo_falla"])
    )

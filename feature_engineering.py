import numpy as np
import pandas as pd


def make_daily_aggregates(df):
    """Devuelve un DataFrame agregado por día, host y tipo_falla con conteo de fallas."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['hora_inicio']).dt.floor('D')
    agg = df.groupby(['host', 'tipo_falla', 'host_id', 'tipo_id', 'date']).size().reset_index(name='n_fallas')
    return agg


def build_time_series_for_combo(agg_df, host, tipo, full_range_min=None, full_range_max=None):
    """
    Construye la serie diaria 0/1 para un host-tipo y retorna DataFrame con fechas y n_fallas.
    Si full_range_min/max son None usa min/max del grupo.
    """
    g = agg_df[(agg_df['host'] == host) & (agg_df['tipo_falla'] == tipo)].sort_values('date')
    if g.empty:
        return pd.DataFrame()
    if full_range_min is None:
        full_range_min = g['date'].min()
    if full_range_max is None:
        full_range_max = g['date'].max()
    dr = pd.date_range(full_range_min, full_range_max, freq='D')
    s = pd.Series(0, index=dr)
    counts = g.set_index('date')['n_fallas']
    s.loc[counts.index] = counts.values
    df = pd.DataFrame({'date': dr, 'n_fallas': s.values})
    df['y'] = (df['n_fallas'] > 0).astype(int)
    return df


def rolling_features(df_ts):
    """Agrega features rolling al DataFrame de serie temporal."""
    df = df_ts.copy()
    df['sum_7'] = df['y'].rolling(7, min_periods=1).sum().values
    df['sum_30'] = df['y'].rolling(30, min_periods=1).sum().values
    df['sum_90'] = df['y'].rolling(90, min_periods=1).sum().values
    df['sum_180'] = df['y'].rolling(180, min_periods=1).sum().values
    df['dow'] = df['date'].dt.weekday
    df['day_of_month'] = df['date'].dt.day
    return df


def make_windows_from_series(df_ts, window=180):
    """
    Genera ventanas (X_seq, X_static, y) a partir del df con rolling features.
    Retorna listas: seqs (window,1), statics, y_targets, dates (target date)
    """
    seqs, statics, ys, dates = [], [], [], []
    for i in range(window, len(df_ts) - 1):
        seq = df_ts['y'].values[i - window:i].reshape(window, 1)
        static = df_ts.loc[i, ['sum_7', 'sum_30', 'sum_90', 'sum_180', 'dow', 'day_of_month']].values.astype(np.float32)
        y = int(df_ts.loc[i + 1, 'y'])
        target_date = df_ts.loc[i + 1, 'date']
        seqs.append(seq.astype(np.float32))
        statics.append(static)
        ys.append(y)
        dates.append(target_date)
    return seqs, statics, ys, dates

import pandas as pd
import numpy as np

def make_daily_aggregates(df):
    df["date"] = pd.to_datetime(df["hora_inicio"]).dt.date
    agg = df.groupby(['host', 'tipo_falla', 'date']).size().reset_index(name='n_fallas')
    return agg

def build_time_series_for_combo(daily, host, tipo, ventana=180):
    """
    Construye secuencias temporales para un host/tipo específico.
    """
    subset = daily[(daily["host"] == host) & (daily["tipo_falla"] == tipo)].copy()
    if subset.empty:
        return [], [], [], []

    subset = subset.set_index("date").asfreq("D", fill_value=0).reset_index()
    seqs, statics, ys, dates = [], [], [], []

    values = subset["n_fallas"].values
    for i in range(len(values) - ventana):
        seqs.append(values[i:i+ventana].reshape(-1, 1))
        statics.append([0])  # Placeholder estático
        ys.append(values[i+ventana])
        dates.append(subset["date"].iloc[i+ventana])

    return np.array(seqs), np.array(statics), np.array(ys), np.array(dates)

def preparar_series(df, host, tipo, ventana=180):
    """
    Wrapper que integra agregación diaria y construcción de secuencias.
    """
    daily = make_daily_aggregates(df)
    return build_time_series_for_combo(daily, host, tipo, ventana)

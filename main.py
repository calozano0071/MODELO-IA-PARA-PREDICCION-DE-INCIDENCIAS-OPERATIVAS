import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import cargar_datos
from feature_engineering import (
    make_daily_aggregates,
    build_time_series_for_combo,
    rolling_features,
    make_windows_from_series
)
from trainer import entrenar_y_evaluar
from utils import set_seed




def main(args):
    print(f"📂 Cargando datos desde: {args.excel}")
    df = cargar_datos(args.excel)
    print(f"Filas totales: {len(df)} - Hosts: {df['host'].nunique()} - Tipos: {df['tipo_falla'].nunique()}")

    # Agregación diaria
    agg = make_daily_aggregates(df)
    combos = agg.groupby(["host", "tipo_falla"])
    print(f"✅ Agregados diarios creados. Combinaciones host-tipo: {len(combos)}")

    print("⚙️ Generando ventanas por combo (puede tardar)...")

    # Acumuladores globales
    X_all, y_all = [], []

    for (host, tipo), grupo in combos:
        X, y = make_windows_from_series(grupo, args.window)

        # Si no hay ventanas, saltar
        if X is None or len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

    # Concatenar todos los datos
    if not X_all:
        print("❌ No se generaron ventanas, revisa el preprocesamiento.")
        return

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"✅ Ventanas creadas: {X_all.shape} - Etiquetas: {y_all.shape}")

    # Dividir en train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=args.val_frac + args.test_frac, shuffle=True
    )
    val_size = args.val_frac / (args.val_frac + args.test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_size, shuffle=True
    )

    # Entrenar modelo
    print("🚀 Entrenando modelo...")
    model, history = entrenar_modelo(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        out_dir=args.out_dir
    )

    # Evaluación final
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Evaluación final -> Loss: {loss:.4f} - Acc: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Ruta del archivo Excel con los datos históricos")
    parser.add_argument("--window", type=int, default=30, help="Tamaño de ventana para series temporales")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--out_dir", type=str, default="models", help="Directorio de salida para guardar modelos")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fracción para validación")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Fracción para test")
    parser.add_argument("--patience", type=int, default=5, help="Patience para early stopping")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de decisión para clasificación")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)

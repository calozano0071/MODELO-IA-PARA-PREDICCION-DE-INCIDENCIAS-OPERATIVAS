import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import cargar_datos, encode_labels
from feature_engineering import preparar_series
from trainer import entrenar_y_evaluar
from utils import set_seed


def main(args):
    print(f"📂 Cargando datos desde: {args.excel}")
    df = cargar_datos(args.excel)
    print(f"Filas totales: {len(df)} - Hosts: {df['host'].nunique()} - Tipos: {df['tipo_falla'].nunique()}")

    # Codificar hosts y tipos a IDs numéricos
    df, host_encoder, tipo_encoder = encode_labels(df)
    n_hosts = len(host_encoder.classes_)
    n_tipos = len(tipo_encoder.classes_)

    # Preparar series (ventanas, features estáticas y etiquetas multi-horizonte)
    print("⚙️ Preparando series temporales...")
    X_seq, X_static, y_day, y_week, y_month, dates, meta = preparar_series(df, window=args.window)

    # Inputs adicionales (IDs de host/tipo)
    host_ids = meta["host"].map({h: i for i, h in enumerate(host_encoder.classes_)}).values
    tipo_ids = meta["tipo_falla"].map({t: i for i, t in enumerate(tipo_encoder.classes_)}).values

    # Dividir en train/val/test
    X_train_idx, X_temp_idx, _, _ = train_test_split(
        np.arange(len(X_seq)), y_day, test_size=args.val_frac + args.test_frac, shuffle=True
    )
    val_size = args.val_frac / (args.val_frac + args.test_frac)
    X_val_idx, X_test_idx, _, _ = train_test_split(
        X_temp_idx, y_day[X_temp_idx], test_size=1 - val_size, shuffle=True
    )

    def build_dataset(idxs):
        return {
            "seq_in": X_seq[idxs],
            "static_in": X_static[idxs],
            "host_in": host_ids[idxs],
            "tipo_in": tipo_ids[idxs]
        }, {
            "out_day": y_day[idxs],
            "out_week": y_week[idxs],
            "out_month": y_month[idxs]
        }

    Xs_train, y_train = build_dataset(X_train_idx)
    Xs_val, y_val = build_dataset(X_val_idx)
    Xs_test, y_test = build_dataset(X_test_idx)

    print(f"✅ Datasets creados -> Train: {len(X_train_idx)} | Val: {len(X_val_idx)} | Test: {len(X_test_idx)}")

    # Entrenar modelo
    print("🚀 Entrenando modelo...")
    model, metrics = entrenar_y_evaluar(
        Xs_train, y_train,
        Xs_val, y_val,
        Xs_test, y_test,
        n_hosts=n_hosts,
        n_tipos=n_tipos,
        output_dir=args.out_dir,
        patience=args.patience,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("📊 Evaluación final en test:")
    print(metrics)


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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)
    main(args)

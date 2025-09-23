import argparse
import os
import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix

from data_loader import cargar_datos, encode_labels
from feature_engineering import make_daily_aggregates, build_time_series_for_combo, rolling_features, make_windows_from_series
from model_lstm import build_lstm_model
from evaluation import evaluate_model, plot_confusion_matrix
from vusalization import plot_loss, plot_auc
from utils import ensure_dir, set_seed

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def main(args):
    set_seed(42)

    # 1) Cargar y preparar datos
    print("Cargando datos desde:", args.excel)
    df = cargar_datos(args.excel)
    df, le_host, le_tipo = encode_labels(df, out_dir=args.out_dir)  # guarda encoders en out_dir
    print(f"Filas totales: {len(df)} - Hosts: {len(le_host.classes_)} - Tipos: {len(le_tipo.classes_)}")

    # 2) Agregados diarios
    agg = make_daily_aggregates(df)
    print("Agregados diarios creados. Combinaciones host-tipo:", len(agg[['host','tipo_falla']].drop_duplicates()))

    # 3) Generar ventanas (seq + static + y) para todos los combos
    all_seqs, all_statics, all_hosts, all_tipos, all_y, all_dates = [], [], [], [], [], []

    combos = agg[['host','tipo_falla']].drop_duplicates()
    print("Generando ventanas por combo (esto puede tardar)...")
    for _, row in combos.iterrows():
        host = row['host']
        tipo = row['tipo_falla']
        ts = build_time_series_for_combo(agg, host, tipo)          # DataFrame con date, n_fallas, y
        if ts.empty:
            continue
        ts = rolling_features(ts)                                 # agrega sum_7,sum_30,...
        seqs, statics, ys, dates = make_windows_from_series(ts, window=args.window)
        if len(seqs) == 0:
            continue
        host_id = int(le_host.transform([host])[0])
        tipo_id = int(le_tipo.transform([tipo])[0])
        for i in range(len(seqs)):
            all_seqs.append(seqs[i])
            all_statics.append(statics[i])
            all_hosts.append(host_id)
            all_tipos.append(tipo_id)
            all_y.append(ys[i])
            all_dates.append(dates[i])

    if len(all_seqs) == 0:
        print("No se generaron ventanas. Revisa window/min_history o cobertura temporal.")
        return

    # 4) Convertir a arrays numpy
    X_seq = np.stack(all_seqs).astype(np.float32)          # (N, window, 1)
    X_static = np.stack(all_statics).astype(np.float32)    # (N, static_dim)
    X_host = np.array(all_hosts).reshape(-1,1).astype(np.int32)
    X_tipo = np.array(all_tipos).reshape(-1,1).astype(np.int32)
    y = np.array(all_y).astype(np.int32)
    dates = np.array(all_dates).astype('datetime64[ns]')

    print("Tamaños: X_seq", X_seq.shape, "X_static", X_static.shape, "X_host", X_host.shape, "y", y.shape)

    # 5) Split temporal (por fechas únicas)
    unique_dates = np.unique(dates)
    n = len(unique_dates)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    test_dates = unique_dates[-n_test:]
    val_dates = unique_dates[-(n_test + n_val):-n_test]
    train_dates = unique_dates[:-(n_test + n_val)]

    train_idx = np.isin(dates, train_dates)
    val_idx = np.isin(dates, val_dates)
    test_idx = np.isin(dates, test_dates)

    print("Fechas train/val/test:", len(train_dates), len(val_dates), len(test_dates))
    print("Ejemplos train/val/test:", train_idx.sum(), val_idx.sum(), test_idx.sum())

    Xs_train = [X_seq[train_idx], X_host[train_idx], X_tipo[train_idx], X_static[train_idx]]
    y_train = y[train_idx]
    Xs_val = [X_seq[val_idx], X_host[val_idx], X_tipo[val_idx], X_static[val_idx]]
    y_val = y[val_idx]
    Xs_test = [X_seq[test_idx], X_host[test_idx], X_tipo[test_idx], X_static[test_idx]]
    y_test = y[test_idx]

    # 6) Class weights (por si hay desbalance)
    classes = np.unique(y_train)
    if len(classes) == 1:
        print("Advertencia: en el set de train sólo hay una clase. Ajusta selección temporal o datos.")
        return
    class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw = {int(classes[i]): float(class_weights_arr[i]) for i in range(len(classes))}
    print("Class weights:", cw)

    # 7) Construir y entrenar modelo
    model = build_lstm_model(window=args.window, static_dim=X_static.shape[1],
                             num_hosts=len(le_host.classes_), num_tipos=len(le_tipo.classes_),
                             emb_host=args.emb_host, emb_tipo=args.emb_tipo)
    model.summary()

    # Callbacks
    ensure_dir(args.out_dir)
    ckpt_path = os.path.join(args.out_dir, 'best_model.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("Entrenando modelo...")
    history = model.fit(
        x=Xs_train,
        y=y_train,
        validation_data=(Xs_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2
    )

    # 8) Evaluación final sobre test
    print("Evaluando sobre test set...")
    y_pred_prob = model.predict(Xs_test, batch_size=args.batch_size)[:,0]
    y_pred = (y_pred_prob >= args.threshold).astype(int)

    # métricas
    auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else float('nan')
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("AUC:", auc)
    print("Precision:", prec, "Recall:", recall, "F1:", f1)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))

    # guardar artefactos
    model.save(os.path.join(args.out_dir, 'final_model.h5'))
    joblib.dump(le_host, os.path.join(args.out_dir, 'le_host.joblib'))
    joblib.dump(le_tipo, os.path.join(args.out_dir, 'le_tipo.joblib'))
    # guardar history y métricas
    joblib.dump(history.history, os.path.join(args.out_dir, 'history.joblib'))
    joblib.dump({'auc': float(auc), 'precision': float(prec), 'recall': float(recall), 'f1': float(f1), 'confusion_matrix': cm.tolist()},
                os.path.join(args.out_dir, 'metrics.joblib'))

    # graficas (opcional)
    plot_loss(history, out_path=os.path.join(args.out_dir, 'loss.png'))
    plot_auc(history, out_path=os.path.join(args.out_dir, 'auc.png'))
    plot_confusion_matrix(cm, labels=[0,1], out_path=os.path.join(args.out_dir, 'confusion.png'))

    print("Todo listo. Modelos y artefactos guardados en:", args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, required=True, help='Ruta al Excel unificado')
    parser.add_argument('--window', type=int, default=180, help='Tamaño de ventana (días)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default='model_output')
    parser.add_argument('--val_frac', type=float, default=0.10)
    parser.add_argument('--test_frac', type=float, default=0.10)
    parser.add_argument('--emb_host', type=int, default=16)
    parser.add_argument('--emb_tipo', type=int, default=8)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
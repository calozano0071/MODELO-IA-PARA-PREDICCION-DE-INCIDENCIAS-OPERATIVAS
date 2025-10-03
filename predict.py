import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

from data_loader import cargar_datos, preparar_datos_prediccion


def main(args):
    print(f"📂 Usando archivo: {args.excel}")

    # ✅ Cargar datos
    df = cargar_datos(args.excel)

    # ✅ Cargar encoders
    le_host = joblib.load("models/le_host.pkl")
    le_tipo = joblib.load("models/le_tipo.pkl")

    # ✅ Preparar datos para predicción
    X_seq, X_static, X_host, X_tipo, meta = preparar_datos_prediccion(
        df, le_host, le_tipo, ventana=args.ventana
    )

    if len(X_seq) == 0:
        print("⚠️ No hay datos suficientes para generar predicciones.")
        return

    # ✅ Cargar modelo multitarea
    model = load_model("models/best_model.keras")
    print("✅ Modelo cargado.")

    # ============================
    # 🔹 Predicciones históricas
    # ============================
    preds_day, preds_week, preds_month = model.predict(
        {"seq_in": X_seq, "static_in": X_static, "host_in": X_host, "tipo_in": X_tipo},
        verbose=0
    )

    meta["pred_day"] = preds_day.flatten()
    meta["pred_week"] = preds_week.flatten()
    meta["pred_month"] = preds_month.flatten()
    meta["es_futuro"] = False

    print("\n📊 Últimas 10 predicciones históricas:")
    print(meta.tail(10)[["fecha", "host", "tipo", "pred_day", "pred_week", "pred_month"]])

    # ============================
    # 🔹 Predicciones futuras
    # ============================
    print(f"\n🔮 Prediciendo {args.horizonte} días hacia adelante...")

    last_seq = X_seq[-1].copy()
    last_static = X_static[-1].copy()
    last_host = X_host[-1]
    last_tipo = X_tipo[-1]
    last_date = pd.to_datetime(meta["fecha"].iloc[-1])

    future_records = []

    for i in range(args.horizonte):
        preds = model.predict(
            {
                "seq_in": last_seq[np.newaxis, ...],
                "static_in": last_static[np.newaxis, ...],
                "host_in": last_host[np.newaxis, ...],
                "tipo_in": last_tipo[np.newaxis, ...],
            },
            verbose=0
        )

        p_day, p_week, p_month = [p[0, 0] for p in preds]
        next_date = last_date + pd.Timedelta(days=i + 1)

        try:
            host_decoded = le_host.inverse_transform(last_host)[0]
        except Exception:
            host_decoded = "UNKNOWN"

        try:
            tipo_decoded = le_tipo.inverse_transform(last_tipo)[0]
        except Exception:
            tipo_decoded = "UNKNOWN"

        future_records.append({
            "fecha": next_date,
            "host": host_decoded,
            "tipo": tipo_decoded,
            "pred_day": p_day,
            "pred_week": p_week,
            "pred_month": p_month,
            "es_futuro": True
        })

        # 🔄 Actualizar secuencia (por ahora solo target diario)
        new_seq = np.roll(last_seq, -1, axis=0)
        new_seq[-1, 0] = p_day   # ⚠️ Asumimos que feature 0 es el target
        last_seq = new_seq

    future_df = pd.DataFrame(future_records)
    print("\n✅ Predicciones futuras generadas.")
    print(future_df.head())

    # ============================
    # 🔹 Guardar resultados
    # ============================
    out_file = "predicciones.csv"
    all_results = pd.concat([meta, future_df], ignore_index=True)
    all_results.to_csv(out_file, index=False)
    print(f"\n💾 Resultados guardados en {out_file}")

    # ============================
    # 🔹 Graficar resultados
    # ============================
    for (h, t), grupo in all_results.groupby(["host", "tipo"]):
        plt.figure(figsize=(12, 6))
        plt.plot(grupo["fecha"], grupo["pred_day"], label="Predicción Día", color="blue")
        plt.plot(grupo["fecha"], grupo["pred_week"], label="Predicción Semana", color="orange")
        plt.plot(grupo["fecha"], grupo["pred_month"], label="Predicción Mes", color="green")

        # 🔹 Si hay valores reales en histórico, graficarlos
        if "target" in grupo.columns:
            plt.plot(grupo["fecha"], grupo["target"], label="Real", color="black", linestyle="dashed")

        plt.title(f"Predicciones para Host={h}, Tipo={t}")
        plt.xlabel("Fecha")
        plt.ylabel("Predicción")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_img = f"pred_{h}_{t}.png"
        plt.savefig(out_img)
        plt.close()
        print(f"📈 Gráfico guardado en {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, required=True)
    parser.add_argument("--ventana", type=int, default=180)
    parser.add_argument("--horizonte", type=int, default=30)
    args = parser.parse_args()

    main(args)

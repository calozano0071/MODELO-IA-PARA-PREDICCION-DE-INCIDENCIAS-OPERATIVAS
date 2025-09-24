import argparse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data_loader import preparar_datos_prediccion

def cargar_modelo_y_encoders():
    """Carga modelo y encoders guardados en model_output/"""
    model = load_model("model_output/final_model.h5")
    le_host = joblib.load("model_output/le_host.joblib")
    le_tipo = joblib.load("model_output/le_tipo.joblib")
    return model, le_host, le_tipo

def predecir(model, X_seq, X_static, X_host, threshold=0.5):
    """Ejecuta predicciones y retorna etiquetas y probabilidades"""
    probs = model.predict([X_seq, X_static, X_host], verbose=0).flatten()
    preds = (probs >= threshold).astype(int)
    return probs, preds

def main(args):
    print(f"Cargando archivo: {args.excel}")
    df = pd.read_excel(args.excel)

    # --- Cargar modelo y encoders ---
    model, le_host, le_tipo = cargar_modelo_y_encoders()

    # --- Preparar datos (mismo pipeline que en entrenamiento) ---
    print("Preparando datos para predicción...")
    X_seq, X_static, X_host, meta = preparar_datos_prediccion(
        df, le_host=le_host, le_tipo=le_tipo, ventana=args.ventana
    )

    # --- Generar predicciones ---
    print("Generando predicciones...")
    probs, preds = predecir(model, X_seq, X_static, X_host, threshold=args.threshold)

    # --- Guardar resultados ---
    resultados = meta.copy()
    resultados["prob_falla"] = probs
    resultados["prediccion"] = preds

    out_file = "predicciones.xlsx"
    resultados.to_excel(out_file, index=False)
    print(f"\n✅ Predicciones guardadas en: {out_file}")
    print(resultados.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicciones con modelo entrenado")
    parser.add_argument("--excel", type=str, required=True, help="Archivo Excel con nuevos datos")
    parser.add_argument("--ventana", type=int, default=180, help="Tamaño de ventana usado en entrenamiento")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de clasificación (0-1)")
    args = parser.parse_args()
    main(args)

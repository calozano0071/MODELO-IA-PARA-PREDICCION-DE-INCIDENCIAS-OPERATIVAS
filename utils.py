import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def ensure_dir(path):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed=42):
    """Fija semillas para reproducibilidad en numpy, random y tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def plot_training_history(history, out_dir="plots", name="training_history.png"):
    """Grafica la evolución de pérdida y métrica durante el entrenamiento."""
    ensure_dir(out_dir)

    plt.figure(figsize=(10, 4))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Evolución de la pérdida")

    # AUC
    if "auc" in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history["auc"], label="train_auc")
        plt.plot(history.history["val_auc"], label="val_auc")
        plt.legend()
        plt.title("Evolución del AUC")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name))
    plt.close()

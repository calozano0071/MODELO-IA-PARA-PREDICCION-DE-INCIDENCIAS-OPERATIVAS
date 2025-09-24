import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

# ===============================
#  MODELO BASE (LSTM + embeddings)
# ===============================
def model_fn(input_shapes, n_hosts, n_tipos):
    """
    Construye el modelo de predicción de fallas.
    
    Parámetros
    ----------
    input_shapes : dict
        Diccionario con shapes de entradas: 
        {"seq_in": (ventana, 1), "static_in": (n_features,)}
    n_hosts : int
        Número de hosts únicos
    n_tipos : int
        Número de tipos de falla únicos
    """
    # Entradas
    seq_in = keras.Input(shape=input_shapes["seq_in"], name="seq_in")
    static_in = keras.Input(shape=input_shapes["static_in"], name="static_in")
    host_in = keras.Input(shape=(1,), name="host_in")
    tipo_in = keras.Input(shape=(1,), name="tipo_in")

    # Embeddings
    emb_host = layers.Embedding(input_dim=n_hosts, output_dim=16, name="emb_host")(host_in)
    emb_tipo = layers.Embedding(input_dim=n_tipos, output_dim=8, name="emb_tipo")(tipo_in)

    # Procesar secuencia
    x_seq = layers.LSTM(64)(seq_in)

    # Flatten embeddings
    x_host = layers.Flatten()(emb_host)
    x_tipo = layers.Flatten()(emb_tipo)

    # Concatenar todo
    x = layers.Concatenate()([x_seq, x_host, x_tipo, static_in])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(
        inputs={"seq_in": seq_in, "static_in": static_in, "host_in": host_in, "tipo_in": tipo_in},
        outputs=out
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="AUC")]
    )
    return model


# ===================================
#  FUNCIÓN DE ENTRENAMIENTO + EVAL
# ===================================
def entrenar_y_evaluar(Xs_train, y_train,
                       Xs_val, y_val,
                       Xs_test, y_test,
                       n_hosts, n_tipos,
                       output_dir="model_output",
                       use_class_weights=True,
                       patience=5,
                       threshold=0.5,
                       epochs=30,
                       batch_size=128):
    """
    Entrena y evalúa el modelo con train/val/test.
    Guarda el mejor modelo en output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calcular class weights si aplica
    class_weights = None
    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, weights))
        print(f"Class weights: {class_weights}")

    # ✅ Usamos las keys del diccionario en vez de índices
    input_shapes = {
        "seq_in": Xs_train["seq_in"].shape[1:],       # (ventana, 1)
        "static_in": Xs_train["static_in"].shape[1:]  # (n_features,)
    }
    model = model_fn(input_shapes, n_hosts, n_tipos)

    # Callbacks
    ckpt = keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "best_model.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    early = keras.callbacks.EarlyStopping(
        patience=patience,
        restore_best_weights=True
    )

    # Entrenar
    model.fit(
        Xs_train,
        y_train,
        validation_data=(Xs_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, early],
        class_weight=class_weights
    )

    # Evaluar en test
    print("Evaluando sobre test set...")
    test_metrics = model.evaluate(Xs_test, y_test, verbose=1)

    return model, test_metrics

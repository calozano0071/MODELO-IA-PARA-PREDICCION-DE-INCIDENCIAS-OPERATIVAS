import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_model(seq_len, n_hosts, n_tipos, out_dir="models", lr=1e-3):
    """
    Construye un modelo multitarea que predice el tipo de falla por día, semana y mes.
    """
    # Entrada secuencial (ventanas históricas)
    seq_in = layers.Input(shape=(seq_len, 1), name="seq_in")
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(seq_in)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Entrada estática (features agregadas)
    static_in = layers.Input(shape=(6,), name="static_in")
    s = layers.Dense(32, activation="relu")(static_in)

    # Entrada de host (embedding)
    host_in = layers.Input(shape=(1,), name="host_in")
    h = layers.Embedding(input_dim=n_hosts + 1, output_dim=16)(host_in)
    h = layers.Flatten()(h)

    # Entrada de tipo_falla (embedding, opcional: "condición")
    tipo_in = layers.Input(shape=(1,), name="tipo_in")
    t = layers.Embedding(input_dim=n_tipos + 1, output_dim=8)(tipo_in)
    t = layers.Flatten()(t)

    # Concatenación de todas las entradas
    concat = layers.Concatenate()([x, s, h, t])
    base = layers.Dense(128, activation="relu")(concat)
    base = layers.Dropout(0.3)(base)

    # Salidas múltiples (multiclase softmax)
    out_day = layers.Dense(n_tipos, activation="softmax", name="out_day")(base)
    out_week = layers.Dense(n_tipos, activation="softmax", name="out_week")(base)
    out_month = layers.Dense(n_tipos, activation="softmax", name="out_month")(base)

    model = models.Model(
        inputs=[seq_in, static_in, host_in, tipo_in],
        outputs=[out_day, out_week, out_month]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "out_day": "sparse_categorical_crossentropy",
            "out_week": "sparse_categorical_crossentropy",
            "out_month": "sparse_categorical_crossentropy",
        },
        metrics={
            "out_day": ["accuracy"],
            "out_week": ["accuracy"],
            "out_month": ["accuracy"],
        }
    )

    model.summary()

    os.makedirs(out_dir, exist_ok=True)
    return model


def entrenar_y_evaluar(
    Xs_train, y_train,
    Xs_val, y_val,
    Xs_test, y_test,
    n_hosts, n_tipos,
    output_dir="models",
    use_class_weights=True,
    patience=6,
    threshold=0.5,
    epochs=30,
    batch_size=128
):
    """
    Entrena y evalúa el modelo multitarea (multiclase).
    """
    model = build_model(
        seq_len=Xs_train["seq_in"].shape[1],
        n_hosts=n_hosts,
        n_tipos=n_tipos,
        out_dir=output_dir
    )

    # Guardado y early stopping
    cb = [
        callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )
    ]

    # Entrenamiento
    history = model.fit(
        Xs_train,
        y_train,
        validation_data=(Xs_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )

    # Evaluación
    metrics = model.evaluate(Xs_test, y_test, verbose=1)

    return model, metrics

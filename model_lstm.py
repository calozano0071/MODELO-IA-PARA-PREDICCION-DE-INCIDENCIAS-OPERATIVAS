import os
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_lstm_model(window, static_dim, num_hosts, num_tipos, emb_host=16, emb_tipo=8):
    """
    Construye un modelo LSTM con:
    - secuencia temporal (LSTM)
    - embeddings de host y tipo de falla
    - features estáticas (rolling features, calendario, etc.)
    """
    # Entradas
    seq_in = Input(shape=(window, 1), name='seq_in')
    static_in = Input(shape=(static_dim,), name='static_in')
    host_in = Input(shape=(1,), name='host_in')
    tipo_in = Input(shape=(1,), name='tipo_in')

    # Embeddings para host y tipo de falla
    emb_h = Embedding(input_dim=num_hosts, output_dim=emb_host, name='emb_host')(host_in)
    emb_h = Flatten()(emb_h)
    emb_t = Embedding(input_dim=num_tipos, output_dim=emb_tipo, name='emb_tipo')(tipo_in)
    emb_t = Flatten()(emb_t)

    # Rama secuencial
    x = LSTM(64, name='lstm')(seq_in)

    # Concatenación de todas las ramas
    x = Concatenate()([x, emb_h, emb_t, static_in])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    # Modelo final
    model = Model(inputs=[seq_in, host_in, tipo_in, static_in], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, out_dir='model_output', epochs=30, batch_size=128):
    """
    Entrena el modelo con EarlyStopping y guarda el mejor y el final.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, 'best_model.h5')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=None, callbacks=callbacks
    )

    model.save(os.path.join(out_dir, 'final_model.h5'))
    return history

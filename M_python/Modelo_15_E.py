# -*- coding: utf-8 -*-
# Script generado automáticamente a partir del notebook Modelo_15.ipynb

def main():
    import os
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, ConvLSTM2D, Conv2D, TimeDistributed,
        Add, Activation, BatchNormalization, Dropout
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
    from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    import matplotlib.pyplot as plt

    # =====================================================
    # CONFIGURACIÓN DE RUTAS
    # =====================================================

    d_original = 'C:/Users/jhon.jaramilloe/Documents/bandas/13/ene_may_5m'
    d_reducida = 'C:/Users/jhon.jaramilloe/Documents/bandas/13/ene_may_5m_reducida'

    if not os.path.exists(d_reducida):
        os.makedirs(d_reducida)

    # =====================================================
    # FUNCIONES AUXILIARES
    # =====================================================

    def resize_npy_images(d_original, d_reducida, new_size=(480, 480)):
        print(f"Redimensionando imágenes en {d_original}...")

        if not os.path.exists(d_reducida):
            os.makedirs(d_reducida)

        processed_count = 0
        for filename in os.listdir(d_original):
            if filename.lower().endswith('.npy'):
                img_path = os.path.join(d_original, filename)
                img_array = np.load(img_path)
                img_resized = np.array(Image.fromarray(img_array).resize(new_size, Image.LANCZOS))
                output_path = os.path.join(d_reducida, filename)
                np.save(output_path, img_resized)
                processed_count += 1

        print(f"Procesadas {processed_count} imágenes")
        return processed_count

    def mostrar_resultados(X, y, y_pred, n=5, cmap='viridis',
                           titulo_general="Comparación de Resultados",
                           xlabel="Eje X (pixeles)", ylabel="Eje Y (pixeles)"):
        n = min(n, len(X))

        fig, axs = plt.subplots(n, 3, figsize=(12, 3 * n))
        fig.suptitle(titulo_general, fontsize=16, y=1.02)

        for i in range(n):
            axs[i, 0].imshow(X[i, ..., 0], cmap=cmap)
            axs[i, 0].set_title(f'Entrada t={i}')
            axs[i, 0].set_xlabel(xlabel)
            axs[i, 0].set_ylabel(ylabel)

            axs[i, 1].imshow(y[i, ..., 0], cmap=cmap)
            axs[i, 1].set_title(f'Real t+1={i+1}')
            axs[i, 1].set_xlabel(xlabel)
            axs[i, 1].set_ylabel(ylabel)

            axs[i, 2].imshow(y_pred[i, ..., 0], cmap=cmap)
            axs[i, 2].set_title(f'Predicción t+1={i+1}')
            axs[i, 2].set_xlabel(xlabel)
            axs[i, 2].set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()

    # =====================================================
    # PREPROCESAMIENTO DE DATOS
    # =====================================================

    print("=" * 50)
    print("PREPROCESAMIENTO DE IMÁGENES")
    print("=" * 50)

    # Verificar y redimensionar imágenes si es necesario
    print("Verificando si necesitas redimensionar imágenes...")
    existing_files = len([f for f in os.listdir(d_reducida) if f.endswith('.npy')])
    if existing_files == 0:
        processed_count = resize_npy_images(d_original, d_reducida, new_size=(480, 480))
        print(f"✅ Redimensionadas {processed_count} imágenes")
    else:
        print(f"✅ Ya existen {existing_files} imágenes redimensionadas en {d_reducida}")

    # Cargar imágenes
    file_list = sorted([f for f in os.listdir(d_reducida) if f.endswith('.npy')])
    print(f"\nCargando {len(file_list)} imágenes desde {d_reducida}...")

    images = []
    for i, filename in enumerate(file_list):
        img_path = os.path.join(d_reducida, filename)
        img_array = np.load(img_path)
        images.append(img_array)

        if (i + 1) % 100 == 0:
            print(f"  Cargadas {i + 1}/{len(file_list)} imágenes...")

    print(f"✅ Carga completada: {len(images)} imágenes")

    # Normalización
    # Normalización
    all_pixels = np.concatenate([img.flatten() for img in images])
    global_min, global_max = np.min(all_pixels), np.max(all_pixels)

    images = [(img - global_min) / (global_max - global_min + 1e-8) for img in images]
    images = [np.clip(img, 0, 1) for img in images]

    # Construir secuencias temporales
    print("\nConstruyendo secuencias temporales...")
    timesteps = 3

    X_base = np.array(images)[..., np.newaxis]
    y_base = X_base.copy()

    N_seq = X_base.shape[0] - timesteps
    X_seq = np.array([X_base[i:i+timesteps] for i in range(N_seq)])
    y_seq = y_base[timesteps:]

    print(f"✅ Secuencias creadas:")
    print(f"   - Forma de X_seq: {X_seq.shape}")
    print(f"   - Forma de y_seq: {y_seq.shape}")
    print(f"   - Número de secuencias: {N_seq}")

    # =====================================================
    # MODELO
    # =====================================================

    print("\n" + "=" * 50)
    print("CONSTRUYENDO MODELO SIMPLIFICADO")
    print("=" * 50)

    # Parámetros
    H, W, C = X_seq.shape[2], X_seq.shape[3], X_seq.shape[4]
    semilla = 42
    tf.random.set_seed(semilla)

    # =====================================================
    # CONSTRUCCIÓN CAPA POR CAPA
    # =====================================================

    # Capa de entrada
    inputs = Input(shape=(timesteps, H, W, C), name='input_sequence')

    # =====================================================
    # BLOQUE 1: Extracción multi-escala inicial
    # =====================================================

    # Rama 3x3 del Bloque 1
    block1_conv3x3 = ConvLSTM2D(
        16, (3, 3), padding='same',
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
        name='block1_conv3x3'
    )(inputs)
    block1_conv3x3_bn = TimeDistributed(BatchNormalization())(block1_conv3x3)
    block1_conv3x3_act = Activation('relu')(block1_conv3x3_bn)

    # Rama 5x5 del Bloque 1
    block1_conv5x5 = ConvLSTM2D(
        16, (5, 5), padding='same',
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
        name='block1_conv5x5'
    )(inputs)
    block1_conv5x5_bn = TimeDistributed(BatchNormalization())(block1_conv5x5)
    block1_conv5x5_act = Activation('relu')(block1_conv5x5_bn)

    # Fusión por SUMA
    block1_merged = Add()([block1_conv3x3_act, block1_conv5x5_act])
    block1_output = Activation('relu', name='block1_output')(block1_merged)
    block1_dropout = Dropout(0.4)(block1_output)

    # =====================================================
    # BLOQUE 2: Con conexión residual
    # =====================================================

    # Guardar para conexión residual
    residual_shortcut = block1_dropout

    # Rama 3x3 del Bloque 2
    block2_conv3x3 = ConvLSTM2D(
        16, (3, 3), padding='same',
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
        name='block2_conv3x3'
    )(block1_dropout)
    block2_conv3x3_bn = TimeDistributed(BatchNormalization())(block2_conv3x3)
    block2_conv3x3_act = Activation('relu')(block2_conv3x3_bn)

    # Rama 5x5 del Bloque 2
    block2_conv5x5 = ConvLSTM2D(
        16, (5, 5), padding='same',
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
        name='block2_conv5x5'
    )(block1_dropout)
    block2_conv5x5_bn = TimeDistributed(BatchNormalization())(block2_conv5x5)
    block2_conv5x5_act = Activation('relu')(block2_conv5x5_bn)

    # Fusión por suma
    block2_merged = Add()([block2_conv3x3_act, block2_conv5x5_act])

    # Conexión RESIDUAL
    block2_residual = Add()([block2_merged, residual_shortcut])
    block2_activated = Activation('relu', name='block2_output')(block2_residual)
    block2_dropout = Dropout(0.4)(block2_activated)

    # =====================================================
    # BLOQUE 3: Procesamiento profundo
    # =====================================================

    # Rama 3x3 del Bloque 3
    block3_conv3x3 = ConvLSTM2D(
        32, (3, 3), padding='same',
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
        name='block3_conv3x3'
    )(block2_dropout)
    block3_conv3x3_bn = TimeDistributed(BatchNormalization())(block3_conv3x3)
    block3_conv3x3_act = Activation('relu')(block3_conv3x3_bn)

    # Rama 5x5 del Bloque 3
    block3_conv5x5 = ConvLSTM2D(
        32, (5, 5), padding='same',
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
        name='block3_conv5x5'
    )(block2_dropout)
    block3_conv5x5_bn = TimeDistributed(BatchNormalization())(block3_conv5x5)
    block3_conv5x5_act = Activation('relu')(block3_conv5x5_bn)

    # Fusión por suma
    block3_merged = Add()([block3_conv3x3_act, block3_conv5x5_act])
    block3_output = Activation('relu', name='block3_output')(block3_merged)
    block3_dropout = Dropout(0.4)(block3_output)

    # =====================================================
    # CAPA FINAL
    # =====================================================

    # Última capa ConvLSTM
    final_convLSTM = ConvLSTM2D(
        16, (3, 3), padding='same',
        return_sequences=False,
        kernel_regularizer=l2(1e-4),
        name='final_convLSTM'
    )(block3_dropout)

    # Capa de salida
    outputs = Conv2D(
        1, (1, 1), padding='same',
        activation='linear',
        name='output'
    )(final_convLSTM)

    # =====================================================
    # CREAR MODELO
    # =====================================================

    model15 = Model(inputs=inputs, outputs=outputs, name='ConvLSTM')

    # Compilar modelo
    initial_lr = 1e-3
    optimizer = Adam(learning_rate=initial_lr)

    model15.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()]
    )

    print("✅ Modelo construido capa por capa:")
    model15.summary()

    # =====================================================
    # CALLBACKS
    # =====================================================

    def cosine_decay_schedule(epoch, lr):
        epochs_total = 50
        min_lr = 1e-5

        if epoch < 10:
            return lr
        else:
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - 10) / (epochs_total - 10)))
            new_lr = min_lr + (initial_lr - min_lr) * cosine_decay
            return new_lr

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath="modelo15_best.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        LearningRateScheduler(cosine_decay_schedule, verbose=1)
    ]

    # =====================================================
    # DIVISIÓN DE DATOS
    # =====================================================

    print("\n" + "=" * 50)
    print("DIVISIÓN DE DATOS")
    print("=" * 50)

    porc_validacion = 0.2
    split_index = int(len(X_seq) * (1 - porc_validacion))

    X_train, X_val = X_seq[:split_index], X_seq[split_index:]
    y_train, y_val = y_seq[:split_index], y_seq[split_index:]

    print(f"✅ Datos divididos:")
    print(f"   - Entrenamiento: {len(X_train)} muestras")
    print(f"   - Validación: {len(X_val)} muestras")

    # =====================================================
    # ENTRENAMIENTO
    # =====================================================

    print("\n" + "=" * 50)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 50)

    history15 = model15.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ ¡Entrenamiento completado!")

    # =====================================================
    # RESULTADOS
    # =====================================================

    print("\n" + "=" * 50)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 50)

    final_train_loss = history15.history['loss'][-1]
    final_val_loss = history15.history['val_loss'][-1]
    best_epoch = np.argmin(history15.history['val_loss']) + 1

    print(f"📊 Resultados finales:")
    print(f"   - Pérdida final entrenamiento: {final_train_loss:.4f}")
    print(f"   - Pérdida final validación: {final_val_loss:.4f}")
    print(f"   - Mejor época: {best_epoch}")

    # =====================================================
    # VISUALIZACIÓN
    # =====================================================

    print("\nGenerando predicciones...")
    y_pred = model15.predict(X_val[:10])

    mostrar_resultados(
        X_val[:10, -1],
        y_val[:10],
        y_pred[:10],
        n=5,
        titulo_general="Modelo 15",
        xlabel="Píxeles (X)",
        ylabel="Píxeles (Y)"
    )

    # Gráficas de entrenamiento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history15.history['loss'], label='Entrenamiento')
    plt.plot(history15.history['val_loss'], label='Validación')
    plt.title('Pérdida - Modelo 15 ')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history15.history['lr'], label='Learning Rate')
    plt.title('Learning Rate durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n🎯 ¡Proceso completado! Modelo guardado: modelo15_best.h5")


    # Extraer métricas del historial (entrenamiento y validación)
    loss = history15.history['loss']
    val_loss = history15.history['val_loss']

    rmse = history15.history['root_mean_squared_error']
    val_rmse = history15.history['val_root_mean_squared_error']

    mape = history15.history['mean_absolute_percentage_error']
    val_mape = history15.history['val_mean_absolute_percentage_error']

    epochs = range(1, len(loss) + 1)

    # Mostrar resumen final
    print("=" * 50)
    print("RESUMEN DE MÉTRICAS")
    print("=" * 50)
    print(f"Entrenamiento - Loss final: {loss[-1]:.4f}, RMSE final: {rmse[-1]:.4f}, MAPE final: {mape[-1]:.4f}")
    print(f"Validación    - Loss final: {val_loss[-1]:.4f}, RMSE final: {val_rmse[-1]:.4f}, MAPE final: {val_mape[-1]:.4f}")
    print("=" * 50)

    # =============================================================================
    # VISUALIZACIÓN DE MÉTRICAS
    # =============================================================================

    plt.figure(figsize=(18, 5))

    # Gráfico 1: Loss (MSE)
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, marker='o', label='Entrenamiento', linewidth=2, markersize=4)
    plt.plot(epochs, val_loss, marker='o', linestyle='--', color='red', linewidth=2, markersize=4, label='Validación')
    plt.title('Loss (MSE) por Época', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Gráfico 2: RMSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, rmse, marker='o', color='orange', label='Entrenamiento', linewidth=2, markersize=4)
    plt.plot(epochs, val_rmse, marker='o', linestyle='--', color='red', linewidth=2, markersize=4, label='Validación')
    plt.title('RMSE por Época', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Gráfico 3: MAPE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, mape, marker='o', color='green', label='Entrenamiento', linewidth=2, markersize=4)
    plt.plot(epochs, val_mape, marker='o', linestyle='--', color='red', linewidth=2, markersize=4, label='Validación')
    plt.title('MAPE por Época', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('MAPE (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle('Comparación de Métricas: Entrenamiento vs Validación - Modelo 15',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Directorio original y reducido para prueba
    d_mar_01_15_2s = 'C:/Users/jhon.jaramilloe/Documents/bandas/13/mar_01_15_2s'
    d_mar_01_15_2s_red =  'C:/Users/jhon.jaramilloe/Documents/bandas/13/mar_01_15_2s_red'

    # Llamar a la función para redimensionar imágenes
    resize_npy_images(d_mar_01_15_2s,d_mar_01_15_2s_red, new_size=(480, 480))

    # Cargar imágenes de prueba
    file_list_test = sorted([f for f in os.listdir(d_mar_01_15_2s_red) if f.endswith('.npy')])
    print(f"Procesando {len(file_list_test)} imágenes para validación")

    images_test = [np.load(os.path.join(d_mar_01_15_2s_red, f)) for f in file_list_test]

    ### corregido
    all_pixels_test = np.concatenate([img.flatten() for img in images_test])
    global_min_test, global_max_test = np.min(all_pixels_test), np.max(all_pixels_test)

    images_test = [(img - global_min_test) / (global_max_test - global_min_test + 1e-8) for img in images_test]
    images_test = [np.clip(img, 0, 1) for img in images_test]

    # Parámetro de secuencia temporal (debe ser el mismo que en entrenamiento)
    timesteps = 3

    # Construir secuencias para validación: X_seq_val (N, timesteps, H, W, C), y_seq_test (N, H, W, C)
    X_base_test = np.array(images_test)[..., np.newaxis]  # (num_imgs_test, H, W, C)
    y_base_test = X_base_test.copy()  # y_base_test igual a X_base_test

    N_seq_test = X_base_test.shape[0] - timesteps
    X_seq_test = np.array([X_base_test[i:i+timesteps] for i in range(N_seq_test)])  # (N_seq_test, timesteps, H, W, C)
    y_seq_test= y_base_test[timesteps:]  # (N_seq_test, H, W, C)

    print(f"Forma de X_seq_test: {X_seq_test.shape}")  # (N_seq_test, timesteps, H, W, C)
    print(f"Forma de y_seq_test: {y_seq_test.shape}")  # (N_seq_test, H, W, C)

    # Ahora puedes evaluar el modelo
    loss, rmse, mape = model15.evaluate(X_seq_test, y_seq_test, verbose=1)
    print(f"Pérdida (MSE): {loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")

    y_pred15_test = model15.predict(X_seq_test)  # Genera las predicciones usando X_seq_val
    mostrar_resultados(
        X_seq_test[:, -1, :, :, :],  # Toma solo el último timestep de la secuencia de entrada
        y_seq_test,                   # Valores reales
        y_pred15_test,                # Predicciones
        n=5,
        titulo_general="Entrada vs Real vs Predicción Modelo 15 - Prueba",
        xlabel="Ancho (px)",
        ylabel="Alto (px)"
    )


if __name__ == '__main__':
    main()

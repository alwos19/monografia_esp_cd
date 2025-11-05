import os
import numpy as np
<<<<<<< HEAD
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import glob

# ==================================================
# CONFIGURACIÓN DE PARÁMETROS
# ==================================================
IMG_HEIGHT = 480
IMG_WIDTH = 480
IMG_CHANNELS = 1
SEQUENCE_LENGTH = 3
BATCH_SIZE = 1
EPOCHS = 50
DROPOUT_RATE = 0.4
VALIDATION_SPLIT = 0.2

# Ruta donde tienes las imágenes ya redimensionadas en formato .npy
IMAGE_DIR = "C:/Users/jhon.jaramilloe/Documents/bandas/13/ene_may_5m_reducida"

# ==================================================
# PREPROCESAMIENTO DE IMÁGENES
# ==================================================
print("==================================================")
print("PREPROCESAMIENTO DE IMÁGENES")
print("==================================================")

# Cargar archivos .npy
print(f"Cargando archivos .npy desde {IMAGE_DIR}...")

# Obtener lista de archivos .npy
npy_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.npy")))
print(f"Encontrados {len(npy_files)} archivos .npy")

# Cargar todos los archivos .npy
images = []
for i, npy_path in enumerate(npy_files):
    try:
        # Cargar directamente el array numpy
        img_array = np.load(npy_path)
        images.append(img_array)
        
        if (i + 1) % 100 == 0:
            print(f"  Cargadas {i + 1}/{len(npy_files)} imágenes...")
            
    except Exception as e:
        print(f"Error cargando {npy_path}: {e}")
        continue

if len(images) == 0:
    print("❌ No se pudieron cargar archivos .npy. Verifica la ruta.")
    exit()

images = np.array(images)
print(f"✅ Carga completada: {len(images)} imágenes")
print(f"   - Forma de images: {images.shape}")

# Normalizar imágenes si es necesario (asumiendo que están en rango 0-255 o 0-1)
print("Normalizando imágenes...")
if images.max() > 1.0:
    images = images.astype(np.float32) / 255.0
else:
    images = images.astype(np.float32)

# Asegurar que tengan la dimensión del canal
if len(images.shape) == 3:  # (batch, height, width)
    images = np.expand_dims(images, axis=-1)  # (batch, height, width, 1)

print("✅ Normalización completada")
print(f"   - Forma final: {images.shape}\n")

# Construir secuencias temporales
print("Construyendo secuencias temporales...")
X_seq = []
y_seq = []

for i in range(len(images) - SEQUENCE_LENGTH):
    # Secuencia de entrada: 3 frames consecutivos
    input_sequence = images[i:i + SEQUENCE_LENGTH]
    # Frame objetivo: el siguiente frame
    target_frame = images[i + SEQUENCE_LENGTH]
    
    X_seq.append(input_sequence)
    y_seq.append(target_frame)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("✅ Secuencias creadas:")
print(f"   - Forma de X_seq: {X_seq.shape}")
print(f"   - Forma de y_seq: {y_seq.shape}")
print(f"   - Número de secuencias: {len(X_seq)}\n")

# ==================================================
# CONSTRUCCIÓN DEL MODELO
# ==================================================
print("==================================================")
print("CONSTRUYENDO MODELO")
print("==================================================")

def create_convlstm_model():
    # Capa de entrada
    input_sequence = keras.Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Bloque 1
    x1 = layers.ConvLSTM2D(16, kernel_size=(3, 3), padding='same', 
                          return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001),
                          name='block1_conv3x3')(input_sequence)
    x1 = layers.TimeDistributed(layers.BatchNormalization())(x1)
    x1 = layers.Activation('relu')(x1)
    
    x2 = layers.ConvLSTM2D(16, kernel_size=(5, 5), padding='same',
                          return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001),
                          name='block1_conv5x5')(input_sequence)
    x2 = layers.TimeDistributed(layers.BatchNormalization())(x2)
    x2 = layers.Activation('relu')(x2)
    
    x = layers.Add()([x1, x2])
    x = layers.Activation('relu', name='block1_output')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Bloque 2
    x1 = layers.ConvLSTM2D(16, kernel_size=(3, 3), padding='same',
                          return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001),
                          name='block2_conv3x3')(x)
    x1 = layers.TimeDistributed(layers.BatchNormalization())(x1)
    x1 = layers.Activation('relu')(x1)
    
    x2 = layers.ConvLSTM2D(16, kernel_size=(5, 5), padding='same',
                          return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001),
                          name='block2_conv5x5')(x)
    x2 = layers.TimeDistributed(layers.BatchNormalization())(x2)
    x2 = layers.Activation('relu')(x2)
    
    x_res = layers.Add()([x1, x2])
    x_res = layers.Add()([x_res, x])  # Conexión residual
    x = layers.Activation('relu', name='block2_output')(x_res)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Bloque 3
    x1 = layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='same',
                          return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001),
                          name='block3_conv3x3')(x)
    x1 = layers.TimeDistributed(layers.BatchNormalization())(x1)
    x1 = layers.Activation('relu')(x1)
    
    x2 = layers.ConvLSTM2D(32, kernel_size=(5, 5), padding='same',
                          return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001),
                          name='block3_conv5x5')(x)
    x2 = layers.TimeDistributed(layers.BatchNormalization())(x2)
    x2 = layers.Activation('relu')(x2)
    
    x = layers.Add()([x1, x2])
    x = layers.Activation('relu', name='block3_output')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Capa final ConvLSTM
    x = layers.ConvLSTM2D(16, kernel_size=(3, 3), padding='same',
                         return_sequences=False, name='final_convLSTM')(x)
    
    # Capa de salida
    output = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=input_sequence, outputs=output, name='ConvLSTM_simplificado')
    return model

# Crear y compilar el modelo
model = create_convlstm_model()

# Función de learning rate con decaimiento cosenoidal
def cosine_decay(epoch):
    initial_lr = 0.001
    decay_steps = EPOCHS
    alpha = 0.0
    epoch = min(epoch, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_lr * decayed

lr_scheduler = keras.callbacks.LearningRateScheduler(cosine_decay)

# Compilar el modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=[
        keras.metrics.MeanAbsolutePercentageError(name='mean_absolute_percentage_error'),
        keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
    ]
)

print("✅ Modelo construido")
model.summary()

# ==================================================
# DIVISIÓN DE DATOS
# ==================================================
print("==================================================")
print("DIVISIÓN DE DATOS")
print("==================================================")

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=VALIDATION_SPLIT, random_state=42, shuffle=True
)

print("✅ Datos divididos:")
print(f"   - Entrenamiento: {len(X_train)} muestras")
print(f"   - Validación: {len(X_val)} muestras\n")

# ==================================================
# CALLBACKS
# ==================================================
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'modelo_best.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ==================================================
# ENTRENAMIENTO
# ==================================================
print("==================================================")
print("INICIANDO ENTRENAMIENTO")
print("==================================================")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping, reduce_lr, checkpoint],
=======
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, Conv2D, TimeDistributed,
    Add, Activation, BatchNormalization, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURACIÓN DE RUTAS
# =====================================================

d_reducida = 'C:/Users/jhon.jaramilloe/Documents/bandas/13/ene_may_5m_reducida'

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

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

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, timesteps=3, batch_size=4, shuffle=True):
        self.images = images
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(images) - timesteps)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            # Crear secuencia temporal
            X_seq = self.images[i:i + self.timesteps]
            y_seq = self.images[i + self.timesteps]
            
            X_batch.append(X_seq)
            y_batch.append(y_seq)
        
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Agregar dimensión del canal si es necesario
        if len(X_batch.shape) == 4:
            X_batch = X_batch[..., np.newaxis]
        if len(y_batch.shape) == 3:
            y_batch = y_batch[..., np.newaxis]
            
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# =====================================================
# PREPROCESAMIENTO DE DATOS
# =====================================================

print("=" * 50)
print("PREPROCESAMIENTO DE IMÁGENES")
print("=" * 50)

# Cargar imágenes ya redimensionadas
file_list = sorted([f for f in os.listdir(d_reducida) if f.endswith('.npy')])
print(f"Cargando {len(file_list)} imágenes desde {d_reducida}...")

images = []
for i, filename in enumerate(file_list):
    img_path = os.path.join(d_reducida, filename)
    img_array = np.load(img_path)
    images.append(img_array)

    if (i + 1) % 100 == 0:
        print(f"  Cargadas {i + 1}/{len(file_list)} imágenes...")

print(f"✅ Carga completada: {len(images)} imágenes")

# Normalización
print("\nNormalizando imágenes...")
images = [(img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) for img in images]
images = [np.clip(img, 0, 1) for img in images]
print("✅ Normalización completada")

# Obtener dimensiones de una imagen de muestra
sample_img = images[0]
H, W = sample_img.shape
C = 1  # Canal único (escala de grises)
timesteps = 3

print(f"\nDimensiones de las imágenes: {H}x{W}")
print(f"Número total de imágenes: {len(images)}")
print(f"Secuencias temporales posibles: {len(images) - timesteps}")

# =====================================================
# CREAR GENERADORES DE DATOS
# =====================================================

print("\n" + "=" * 50)
print("CREANDO GENERADORES DE DATOS")
print("=" * 50)

# Dividir índices para entrenamiento y validación
total_sequences = len(images) - timesteps
indices = np.arange(total_sequences)
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

print(f"Secuencias de entrenamiento: {len(train_idx)}")
print(f"Secuencias de validación: {len(val_idx)}")

# Crear generadores
batch_size = 4
train_generator = DataGenerator(images, timesteps=timesteps, batch_size=batch_size, shuffle=True)
val_generator = DataGenerator(images, timesteps=timesteps, batch_size=batch_size, shuffle=False)

# =====================================================
# MODELO SIMPLIFICADO
# =====================================================

print("\n" + "=" * 50)
print("CONSTRUYENDO MODELO SIMPLIFICADO")
print("=" * 50)

# Parámetros
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

model15 = Model(inputs=inputs, outputs=outputs, name='ConvLSTM_simplificado_expandido')

# Compilar modelo
initial_lr = 1e-4
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
    warmup_epochs = 5
    decay_epochs = epochs_total - warmup_epochs
    
    if epoch < warmup_epochs:
        return initial_lr * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / decay_epochs
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        new_lr = initial_lr * cosine_decay
        return max(new_lr, 1e-6)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath="modelo15_best.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    LearningRateScheduler(cosine_decay_schedule, verbose=1)
]

# =====================================================
# ENTRENAMIENTO
# =====================================================

print("\n" + "=" * 50)
print("INICIANDO ENTRENAMIENTO")
print("=" * 50)

history15 = model15.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks,
>>>>>>> model15
    verbose=1
)

print("✅ ¡Entrenamiento completado!")

<<<<<<< HEAD
# ==================================================
# RESULTADOS
# ==================================================
print("==================================================")
print("RESULTADOS DEL ENTRENAMIENTO")
print("==================================================")

# Encontrar la mejor época
best_epoch = np.argmin(history.history['val_loss']) + 1
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("📊 Resultados finales:")
print(f"   - Pérdida final entrenamiento: {final_train_loss:.4f}")
print(f"   - Pérdida final validación: {final_val_loss:.4f}")
print(f"   - Mejor época: {best_epoch}\n")

# ==================================================
# GENERAR PREDICCIONES
# ==================================================
print("Generando predicciones...")
sample_sequence = X_val[0:1]
prediction = model.predict(sample_sequence, verbose=1)

print("✅ Predicción generada")
print(f"   - Forma de la predicción: {prediction.shape}")

# Guardar el modelo final
model.save('modelo_final.h5')
print("✅ Modelo guardado como 'modelo_final.h5'")
=======
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

# Crear un pequeño conjunto de datos para visualización
X_val_vis, y_val_vis = val_generator[0]  # Primer batch de validación

y_pred = model15.predict(X_val_vis)

mostrar_resultados(
    X_val_vis[:, -1],  # Último frame de cada secuencia
    y_val_vis,
    y_pred,
    n=min(5, batch_size),
    titulo_general="Modelo 15 - Arquitectura Simplificada",
    xlabel="Píxeles (X)",
    ylabel="Píxeles (Y)"
)

# Gráficas de entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history15.history['loss'], label='Entrenamiento')
plt.plot(history15.history['val_loss'], label='Validación')
plt.title('Pérdida - Modelo 15')
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
>>>>>>> model15

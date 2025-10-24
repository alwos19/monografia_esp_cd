import os
import numpy as np
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
    verbose=1
)

print("✅ ¡Entrenamiento completado!")

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
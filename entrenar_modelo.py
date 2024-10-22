from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ----- Paso 1: Cargar el dataset con ImageDataGenerator -----

# Definir el directorio del dataset (asegúrate de que la ruta sea correcta)
train_data_dir = 'C:/Users/jdvil/OneDrive/Escritorio/Inteligencia Artificial Eli/datagarbage-classification/archive/Garbage classification/Garbage classification'

# Generador de datos con aumento de imágenes
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

# Cargar las imágenes de entrenamiento y validación
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224),
                                                    batch_size=32, class_mode='categorical', subset='training')

validation_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224),
                                                         batch_size=32, class_mode='categorical', subset='validation')

# ----- Paso 2: Configuración del modelo preentrenado -----

# Cargar el modelo preentrenado MobileNetV2 sin las capas superiores
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas base del modelo
base_model.trainable = False

# Añadir nuevas capas superiores
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)  # 6 clases: cardboard, glass, metal, paper, plastic, trash

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar un resumen del modelo
model.summary()

# ----- Paso 3: Entrenar el modelo -----

# Entrenar el modelo
history = model.fit(train_generator, validation_data=validation_generator, epochs=10, steps_per_epoch=100, validation_steps=50)

# Guardar el modelo
model.save('modelo_clasificacion_residuos.h5')

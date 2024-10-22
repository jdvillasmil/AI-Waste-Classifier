import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
modelo = load_model('modelo_clasificacion_residuos.h5')

# Definir las clases de residuos
clases = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Iniciar la captura de video desde la cámara con índice 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    # Redimensionar la imagen al tamaño que requiere el modelo (224x224)
    imagen = cv2.resize(frame, (224, 224))
    
    # Normalizar la imagen (ya que el modelo fue entrenado con imágenes normalizadas)
    imagen = imagen.astype('float32') / 255.0
    
    # Expandir las dimensiones de la imagen para que se ajuste al input del modelo
    imagen = np.expand_dims(imagen, axis=0)
    
    # Realizar la predicción
    prediccion = modelo.predict(imagen)
    
    # Obtener la clase con mayor probabilidad
    clase_predicha = np.argmax(prediccion)
    clase_residuo = clases[clase_predicha]

    # Mostrar el frame con la predicción
    cv2.putText(frame, f'Residuo: {clase_residuo}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Detección de Residuos', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

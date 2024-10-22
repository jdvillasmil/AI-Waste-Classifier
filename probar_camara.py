import cv2

# Probar varios índices de cámara (puedes ajustar el rango si es necesario)
for i in range(5):  # Intenta con los primeros 5 índices de cámara
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara {i} encontrada y accesible.")
        cap.release()  # Liberar la cámara después de la prueba
    else:
        print(f"Cámara {i} no accesible.")

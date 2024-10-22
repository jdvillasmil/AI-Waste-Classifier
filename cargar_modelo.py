import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import load_model

# Clase para manejar la cámara, la clasificación y la GUI
class WasteClassifierCamera:
    def __init__(self, model_path):
        # Cargar el modelo de clasificación
        self.model = load_model(model_path)

        # Definir las clases de residuos
        self.classes = ['Plástico', 'Vidrio', 'Cartón', 'Metal', 'Orgánico', 'Otro']

        # Configuración de la interfaz de cámara con Tkinter
        self.window = tk.Tk()
        self.window.title("Clasificación de Residuos en Tiempo Real")

        self.video_source = 0  # Índice de la cámara

        # Crear un widget de Label para mostrar el feed de la cámara
        self.vid_label = Label(self.window)
        self.vid_label.grid(row=0, column=0, columnspan=3)

        # Label para mostrar la predicción
        self.result_label = Label(self.window, text="Clasificación: ", font=("Arial", 16))
        self.result_label.grid(row=1, column=0, columnspan=3)

        # Botón para iniciar la cámara
        self.btn_start = Button(self.window, text="Iniciar Cámara", width=20, command=self.start_camera)
        self.btn_start.grid(row=2, column=0)

        # Botón para detener la cámara
        self.btn_stop = Button(self.window, text="Detener Cámara", width=20, command=self.stop_camera)
        self.btn_stop.grid(row=2, column=1)

        # Botón para pausar el video
        self.btn_pause = Button(self.window, text="Pausar", width=20, command=self.pause_camera)
        self.btn_pause.grid(row=2, column=2)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Inicializar captura de video
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        self.running = False  # Para saber si la cámara está en ejecución
        self.paused = False  # Para saber si el video está pausado

        self.window.mainloop()

    def start_camera(self):
        if not self.running:
            self.running = True
            self.update()

    def stop_camera(self):
        self.running = False

    def pause_camera(self):
        if self.paused:
            self.paused = False
            self.btn_pause.config(text="Pausar")
            self.update()  # Volver a actualizar el video
        else:
            self.paused = True
            self.btn_pause.config(text="Reanudar")

    def update(self):
        if self.running and not self.paused:
            ret, frame = self.vid.read()
            if ret:
                # Convertir la imagen de OpenCV (BGR) a PIL (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Mostrar un cuadro verde en el área de enfoque
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                box_size = 100  # Tamaño del cuadro de enfoque
                top_left = (center_x - box_size // 2, center_y - box_size // 2)
                bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                # Recortar el área dentro del cuadro para la clasificación
                cropped_frame = frame_rgb[center_y - box_size//2:center_y + box_size//2,
                                          center_x - box_size//2:center_x + box_size//2]

                # Redimensionar la imagen para el modelo
                cropped_frame_resized = cv2.resize(cropped_frame, (224, 224))  # Cambiar el tamaño según tu modelo
                cropped_frame_resized = np.expand_dims(cropped_frame_resized, axis=0) / 255.0

                # Hacer predicción de clasificación
                prediction = self.model.predict(cropped_frame_resized)
                class_index = np.argmax(prediction)
                classification = self.classes[class_index]

                # Actualizar el Label con la clasificación
                self.result_label.config(text=f"Clasificación: {classification}")

                # Convertir la imagen a PIL y luego a un formato compatible con Tkinter
                image = Image.fromarray(frame_rgb)
                image_tk = ImageTk.PhotoImage(image=image)

                # Mostrar el video en el Label
                self.vid_label.imgtk = image_tk
                self.vid_label.config(image=image_tk)

            # Llamar de nuevo a update después de 10 ms
            self.window.after(10, self.update)

    def on_closing(self):
        self.stop_camera()
        self.vid.release()
        self.window.destroy()

# Ruta al modelo entrenado
model_path = "modelo_clasificacion_residuos.h5"  # Asegúrate de poner la ruta correcta aquí

# Crear la aplicación
app = WasteClassifierCamera(model_path)

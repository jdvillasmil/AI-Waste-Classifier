import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import os
import cargar_modelo  # Importar la lógica de cargar_modelo.py

# Clase para manejar la cámara y la GUI
class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0  # Índice de la cámara

        # Crear un widget de Label para mostrar el feed de la cámara
        self.vid_label = Label(window)
        self.vid_label.grid(row=0, column=0, columnspan=3)

        # Botón para iniciar la cámara
        self.btn_start = Button(window, text="Iniciar Cámara", width=20, command=self.start_camera)
        self.btn_start.grid(row=1, column=0)

        # Botón para detener la cámara
        self.btn_stop = Button(window, text="Detener Cámara", width=20, command=self.stop_camera)
        self.btn_stop.grid(row=1, column=1)

        # Botón para pausar el video
        self.btn_pause = Button(window, text="Pausar", width=20, command=self.pause_camera)
        self.btn_pause.grid(row=1, column=2)

        # Botón para capturar una imagen
        self.btn_capture = Button(window, text="Capturar Imagen", width=20, command=self.capture_and_classify_image)
        self.btn_capture.grid(row=2, column=1)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image_tk = ImageTk.PhotoImage(image=image)

                # Mostrar el video en el Label
                self.vid_label.imgtk = image_tk
                self.vid_label.config(image=image_tk)

            # Llamar de nuevo a update después de 10 ms
            self.window.after(10, self.update)

    def capture_and_classify_image(self):
        if self.running and not self.paused:
            ret, frame = self.vid.read()
            if ret:
                # Guardar la imagen capturada
                if not os.path.exists("capturas"):
                    os.makedirs("capturas")  # Crear el directorio si no existe
                image_path = os.path.join("capturas", "captura.png")
                cv2.imwrite(image_path, frame)  # Guardar la imagen
                print(f"Imagen guardada en: {image_path}")
                
                # Clasificar la imagen usando el modelo cargado
                resultado = cargar_modelo.predecir_clasificacion(image_path)  # Llamada al modelo
                print(f"Resultado de la clasificación: {resultado}")

    def on_closing(self):
        self.stop_camera()
        self.vid.release()
        self.window.destroy()

# Crear la ventana de Tkinter
root = tk.Tk()
app = CameraApp(root, "Cámara con Tkinter - Clasificación de Residuos")

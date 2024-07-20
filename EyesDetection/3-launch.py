import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import threading
import time

# Carica i classificatori Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Carica il modello
model = load_model("C:/Users/39329/Desktop/Progetto CV/Eyes_Model.h5")

#  video capture
cap = cv2.VideoCapture(0)
label = "No Prediction"
running = True


# Funzione per elaborare l'immagine e fare la predizione
def process_frame(frame):
    global label
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

    for (ex, ey, ew, eh) in eyes:
        eye = frame[ey:ey + eh, ex:ex + ew]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(1, 80, 80, 3)

        prediction = model.predict(eye)
        label = "Open" if prediction[0][0] < prediction[0][1] else "Closed"
        break  # Predici solo su un occhio per ridurre il carico di elaborazione


# Funzione per visualizzare il video
def display_video():
    global label, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Disegna il rettangolo nero per il testo
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        # Mostra il risultato della predizione
        cv2.putText(frame, label, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostra il frame
        cv2.imshow('frame', frame)

        #cv2.imshow('eye',eye)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cap.release()
    cv2.destroyAllWindows()


# Esegui l'elaborazione dell'immagine in un thread separato
thread = threading.Thread(target=display_video)
thread.start()

while thread.is_alive():
    ret, frame = cap.read()
    if ret:
        process_frame(frame)
    time.sleep(0.03)  # piccola pausa per ridurre l'uso della CPU

thread.join()



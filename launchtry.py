import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import threading
import time

# Carica i classificatori Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Carica i modelli
eye_model = load_model("C:/Users/39329/Desktop/Progetto CV/Eyes_Model.h5")
keypoint_model = load_model("C:/Users/39329/Desktop/Progetto CV/Keypoint_Mobilenet_Model.h5")

# Variabile globale per il video capture
cap = cv2.VideoCapture(0)
label = "No Prediction"
running = True

# Funzione per elaborare l'immagine e fare la predizione
def process_frame(frame):
    global label
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

    print(f"Faces detected: {len(faces)}")  # Debug: numero di volti rilevati

    for (ex, ey, ew, eh) in eyes:
        eye = frame[ey:ey + eh, ex:ex + ew]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(1, 80, 80, 3)

        prediction = eye_model.predict(eye)
        label = "Open" if prediction[0][0] < prediction[0][1] else "Closed"
        break  # Predici solo su un occhio per ridurre il carico di elaborazione

    # Rileva i keypoint del viso
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (80, 80))  # Ridimensiona per il modello di keypoint
        face_resized = face_resized / 255.0
        face_resized = face_resized.reshape(1, 80, 80, 3)  # Aggiungi la dimensione del canale

        keypoints = keypoint_model.predict(face_resized)[0]
        keypoints = keypoints * 40 + 40  # Ripristina i keypoint alle dimensioni originali
        keypoints = keypoints.reshape(-1, 2)

        print(f"Keypoints (normalized): {keypoints}")  # Messaggio di debug per verificare i keypoint

        # Scala i keypoint alle dimensioni del volto rilevato
        for (kx, ky) in keypoints:
            kx = int(kx * w / 80)
            ky = int(ky * h / 80)
            if 0 <= x + kx < frame.shape[1] and 0 <= y + ky < frame.shape[0]:
                print(f"Drawing keypoint at: {(x + kx, y + ky)}")  # Messaggio di debug per le coordinate scalate
                cv2.circle(frame, (x + kx, y + ky), 3, (0, 255, 0), -1)  # Usa un cerchio più grande e visibile
            else:
                print(f"Keypoint out of bounds: {(x + kx, y + ky)}")

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
    time.sleep(0.03)  # Aggiungi una piccola pausa per ridurre l'uso della CPU

thread.join()

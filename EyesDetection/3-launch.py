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
    time.sleep(0.03)  # Aggiungi una piccola pausa per ridurre l'uso della CPU

thread.join()


# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # Caricamento del modello
# model_path = "C:/Users/39329/Desktop/Progetto CV/drowsiness_model.h5"
# model = load_model(model_path)
#
#
# # Funzione per il pre-processing dell'immagine
# def preprocess_image(image):
#     if image is not None:
#         print("Image shape before preprocessing:", image.shape)
#         # Ridimensiona l'immagine alle stesse dimensioni usate durante l'addestramento
#         image = cv2.resize(image, (64, 64))
#         # Normalizza l'immagine
#         image = image / 255.0
#         # Aggiungi una dimensione in più per il batch (1 immagine)
#         image = np.expand_dims(image, axis=0)
#         print("Preprocessed image shape:", image.shape)
#     return image
#
#
# # Funzione per prevedere se gli occhi sono aperti o chiusi
# def predict_eye_state(image):
#     # Effettua il pre-processing dell'immagine
#     preprocessed_image = preprocess_image(image)
#     print("Preprocessed image:", preprocessed_image)
#
#     # Effettua la previsione utilizzando il modello solo se l'immagine è valida
#     if preprocessed_image is not None:
#         prediction = model.predict(preprocessed_image)
#         print("Prediction:", prediction)
#         # Se la probabilità associata alla classe 1 (occhi chiusi) è maggiore,
#         # restituisci "Closed", altrimenti restituisci "Open"
#         if prediction[0][1] > prediction[0][0]:
#             return "Closed"
#         else:
#             return "Open"
#     else:
#         return "Invalid Image"
#
#
# # Apertura della telecamera
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Impossibile leggere l'immagine dalla telecamera")
#         break
#
#     # Effettua la previsione sugli occhi nel frame
#     eye_state = predict_eye_state(frame)
#     print("Eye State:", eye_state)
#
#     # Visualizza il risultato
#     cv2.putText(frame, f"Eye State: {eye_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow("Frame", frame)
#
#     # Esci dal ciclo se premi 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Rilascia la risorsa della telecamera e chiudi le finestre
# cap.release()
# cv2.destroyAllWindows()

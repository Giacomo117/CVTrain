import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Caricamento del modello
model_path = "C:/Users/39329/Desktop/Progetto CV/drowsiness_model.h5"
model = load_model(model_path)


# Funzione per il pre-processing dell'immagine
def preprocess_image(image):
    if image is not None:
        print("Image shape before preprocessing:", image.shape)
        # Ridimensiona l'immagine alle stesse dimensioni usate durante l'addestramento
        image = cv2.resize(image, (64, 64))
        # Normalizza l'immagine
        image = image / 255.0
        # Aggiungi una dimensione in più per il batch (1 immagine)
        image = np.expand_dims(image, axis=0)
        print("Preprocessed image shape:", image.shape)
    return image


# Funzione per prevedere se gli occhi sono aperti o chiusi
def predict_eye_state(image):
    # Effettua il pre-processing dell'immagine
    preprocessed_image = preprocess_image(image)
    print("Preprocessed image:", preprocessed_image)

    # Effettua la previsione utilizzando il modello solo se l'immagine è valida
    if preprocessed_image is not None:
        prediction = model.predict(preprocessed_image)
        print("Prediction:", prediction)
        # Se la probabilità associata alla classe 1 (occhi chiusi) è maggiore,
        # restituisci "Closed", altrimenti restituisci "Open"
        if prediction[0][1] > prediction[0][0]:
            return "Closed"
        else:
            return "Open"
    else:
        return "Invalid Image"


# Apertura della telecamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossibile leggere l'immagine dalla telecamera")
        break

    # Effettua la previsione sugli occhi nel frame
    eye_state = predict_eye_state(frame)
    print("Eye State:", eye_state)

    # Visualizza il risultato
    cv2.putText(frame, f"Eye State: {eye_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    # Esci dal ciclo se premi 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la risorsa della telecamera e chiudi le finestre
cap.release()
cv2.destroyAllWindows()

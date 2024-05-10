import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carica il modello preaddestrato
model_path = "C:/Users/39329/Desktop/Progetto CV/drowsiness_model.h5"
model = load_model(model_path)

# Funzione per il pre-processing dell'immagine
def preprocess_image(image):
    # Ridimensiona l'immagine alle stesse dimensioni usate durante l'addestramento
    image = cv2.resize(image, (64, 64))
    # Normalizza l'immagine
    image = image / 255.0
    # Aggiungi una dimensione in più per il batch (1 immagine)
    image = np.expand_dims(image, axis=0)
    return image

# Funzione per prevedere se gli occhi sono aperti o chiusi
def predict_eye_state(image):
    # Effettua il pre-processing dell'immagine
    preprocessed_image = preprocess_image(image)

    # Effettua la previsione utilizzando il modello
    prediction = model.predict(preprocessed_image)

    # Se la probabilità associata alla classe 1 (occhi chiusi) è maggiore,
    # allora restituisci "Closed", altrimenti restituisci "Open"
    if prediction[0][0] > prediction[0][1]:
        return "Closed"
    else:
        return "Open"

# Apri l'immagine
#image_path = "C:/Users/39329/Desktop/open.jpeg"
# Apri l'immagine
#image_path = "C:/Users/39329/Desktop/closed.jpg"
#image_path = "C:/Users/39329/Desktop/1.jpg"
#image_path = "C:/Users/39329/Desktop/2.jpg"
#image_path = "C:/Users/39329/Desktop/3.jpg"

image = cv2.imread(image_path)


# Effettua la previsione sugli occhi nell'immagine
eye_state = predict_eye_state(image)

# Visualizza il risultato
cv2.putText(image, f"Eye State: {eye_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Predicted Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

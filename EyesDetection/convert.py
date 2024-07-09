from tensorflow.keras.models import load_model

# Carica il modello da un file .keras
model = load_model("C:/Users/39329/Desktop/Progetto CV/model.keras")

# Salva il modello in formato .h5
model.save("C:/Users/39329/Desktop/Progetto CV/model.h5")

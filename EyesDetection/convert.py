from tensorflow.keras.models import load_model
import torch
# Carica il modello da un file .keras
#model = load_model("C:/Users/39329/Desktop/Progetto CV/second_trained_keypoints_model.pt")

# Salva il modello in formato .h5
#model.save("C:/Users/39329/Downloads/second_trained_keypoints_model.h5")

model = torch.hub.load("C:/Users/39329/Desktop/Progetto CV/", 'custom', path=r"C:/Users/39329/Desktop/Progetto CV/second_trained_keypoints_model.pt", source='local')

torch.save(model, "C:/Users/39329/Downloads/second_trained_keypoints_model.h5")
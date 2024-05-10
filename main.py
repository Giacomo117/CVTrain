import os
import cv2
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D,MaxPooling2D,BatchNormalization, Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

# Definizione della funzione per caricare le immagini e le etichette
print("Inizio esecuzione del codice")


def load_data(data_dir):
    images = []
    labels = []

    # Percorri le cartelle Drowsy e Non_Drowsy
    for label, category in enumerate(['Non_Drowsy', 'Drowsy']):
        folder_path = os.path.join(data_dir, category)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Impossibile caricare l'immagine: {image_path}")
            # else:
            # print(f"Caricata l'immagine: {image_path}")
            image = cv2.resize(image, (64, 64))  # Ridimensiona l'immagine se necessario
            images.append(image)
            labels.append(label)

    # Converte le liste in numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Caricamento dei dati
data_dir = "C:/Users/39329/Desktop/Progetto CV/archive/Driver Drowsiness Dataset (DDD)"
images, labels = load_data(data_dir)

# Normalizzazione delle immagini
images = images / 255.0

# Dividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Bilanciamento delle classi con oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.reshape(-1, 64 * 64 * 3), y_train)
X_train_resampled = X_train_resampled.reshape(-1, 64, 64, 3)


# Definizione del modello CNN
def create_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    model.summary()
    return model


# Addestramento del modello
input_shape = (64, 64, 3)  # Dimensioni delle immagini
cnn_model = create_model(input_shape)
print('Modello creato')

# Compilazione del modello
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Usa sparse_categorical_crossentropy
                  metrics=['accuracy'])

# Addestramento del modello
print('Inizio addestramento')
history = cnn_model.fit(X_train_resampled, y_train_resampled,
                        epochs=5,
                        batch_size=32,
                        validation_data=(X_test, y_test))

# Salvataggio del modello
model_dir = "C:/Users/39329/Desktop/Progetto CV"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "drowsiness_model.h5")
cnn_model.save(model_path)
print("Modello salvato in:", model_path)

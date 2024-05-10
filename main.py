import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, LeakyReLU

# Funzione per caricare le immagini e le etichette

# Percorso del dataset
dataset_dir = r'C:\Users\39329\Desktop\Progetto CV\mrlEyes_2018_01\mrlEyes_2018_01'

print("Inizio esecuzione codice")
def load_dataset(dataset_dir):
    images = []
    labels = []

    for folder in os.listdir(dataset_dir):
        if folder.startswith("s"):  # Ignora eventuali file nascosti
            subject_dir = os.path.join(dataset_dir, folder)
            for image_file in os.listdir(subject_dir):
                if image_file.endswith(".png"):
                    image_path = os.path.join(subject_dir, image_file)
                    image = load_img(image_path, target_size=(84, 84))  # Regola la dimensione dell'immagine come necessario
                    image_array = img_to_array(image)
                    images.append(image_array)

                    # Ottieni le etichette dall'immagine
                    parts = image_file.split('_')
                    gender = int(parts[2])
                    eye_state = int(parts[3])
                    labels.append((gender, eye_state))

    return np.array(images), np.array(labels)



# Carica il dataset
images, labels = load_dataset(dataset_dir)

# Divisione in set di addestramento, test e validazione
X_train, X_test_val, y_train, y_test_val = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

# Stampa delle dimensioni dei set
print("Dimensioni Train Set:", X_train.shape, y_train.shape)
print("Dimensioni Validation Set:", X_val.shape, y_val.shape)
print("Dimensioni Test Set:", X_test.shape, y_test.shape)


# Definizione del modello CNN
def create_model(input_shape):
    model = Sequential()
    ## 001
    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    ## 002
    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## 003
    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    ## 004
    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## 005
    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    ## 006
    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## 007
    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    ## 008
    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## 009
    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    ## 010
    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## 011
    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    ## 012
    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    # MLP
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    model.summary()
    return model


# Addestramento del modello
input_shape = (224, 224, 3)  # Dimensioni delle immagini
cnn_model = create_model(input_shape)
print('Modello creato')

# Compilazione del modello
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Usa sparse_categorical_crossentropy
                  metrics=['accuracy'])

# Addestramento del modello
print('Inizio addestramento')
history = cnn_model.fit(X_train, y_train,
                        epochs=5,
                        batch_size=32,
                        validation_data=(X_val, y_val))

# Salvataggio del modello
model_dir = "C:/Users/39329/Desktop/Progetto CV"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "drowsiness_model.h5")
cnn_model.save(model_path)
print("Modello salvato in:", model_path)

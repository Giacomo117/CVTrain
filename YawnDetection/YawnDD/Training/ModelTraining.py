import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

from tqdm import tqdm


# class CustomDataGen:
#     def __init__(self, directory, batch_size=32, image_size=(64, 64)):
#         self.directory = directory
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.image_filenames = os.listdir(directory)
#         self.index = 0

#     def __next__(self):
#         batch_images = []
#         batch_labels = []

#         for i in range(self.batch_size):
#             if self.index >= len(self.image_filenames):
#                 self.index = 0

#             image_filename = self.image_filenames[self.index]
#             image_path = os.path.join(self.directory, image_filename)
#             image = load_img(image_path, target_size=self.image_size)
#             image = img_to_array(image) / 255.0  # normalize pixel values

#             if "Yawning" in image_filename:
#                 label = [1, 0]  # "Yawning"
#             else:
#                 label = [0, 1]  # "No Yawning"

#             batch_images.append(image)
#             batch_labels.append(label)

#             self.index += 1

#         return np.array(batch_images), np.array(batch_labels)
    
#     def __len__(self):
#         return (len(self.image_filenames) + self.batch_size - 1) // self.batch_size

#     def __iter__(self):
#         return self
print("Inizio dello script")

# Definizione della callback per la barra di avanzamento
class TqdmProgressCallback(Callback):

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Inizio epoca {epoch+1}/{self.epochs}')
        self.pbar = tqdm(total=self.params['steps'], position=0, leave=True)

    def on_batch_end(self, batch, logs=None):
        self.pbar.update()

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()

print("Definizione del modello")
# Definizione del modello
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Primo strato convoluzionale
    MaxPooling2D(pool_size=(2, 2)),  # Primo strato di pooling
    Conv2D(64, (3, 3), activation='relu'),  # Secondo strato convoluzionale
    MaxPooling2D(pool_size=(2, 2)),  # Secondo strato di pooling
    Conv2D(128, (3, 3), activation='relu'),  # Terzo strato convoluzionale
    MaxPooling2D(pool_size=(2, 2)),  # Terzo strato di pooling
    Flatten(),  # Strato per appiattire l'input
    Dense(128, activation='relu'),  # Strato completamente connesso
    Dropout(0.5),  # Strato di dropout per prevenire l'overfitting
    Dense(2, activation='softmax')  # Strato di output
])

print("Compilazione del modello")
# Compilazione del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Generazione dei dati per l'addestramento")
# Generatore di dati per l'addestramento
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Ridimensionamento delle immagini
    shear_range=0.2,  # Range per le trasformazioni di shear
    zoom_range=0.2,  # Range per lo zoom
    horizontal_flip=True,  # Abilita il flip orizzontale
    validation_split=0.2  # Percentuale di dati da usare come validazione
)

print("Generazione dei dati per l'addestramento")
# Generatore di dati per l'addestramento
train_generator = train_datagen.flow_from_directory(
    './annotated_frames',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

print("Generazione dei dati per la validazione")
# Generatore di dati per la validazione
validation_generator = train_datagen.flow_from_directory(
    './annotated_frames',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Addestramento del modello")
# Addestramento del modello
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=25,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[TqdmProgressCallback()]  # Aggiunta della callback per la barra di avanzamento
)

print("Valutazione del modello")
# Valutazione del modello
loss, accuracy = model.evaluate(validation_generator)
print(f"Accuratezza della validazione: {accuracy}")

print("Salvataggio del modello")
# Salvataggio del modello
model.save('yawn_detection_model.h5')

print("Fine dello script")
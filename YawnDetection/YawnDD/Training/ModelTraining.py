import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import numpy as np
from tqdm import tqdm

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

print("Caricamento del modello MobileNet pre-addestrato")
# Caricamento del modello MobileNet pre-addestrato
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(64, 64, 3))  # Assicurati che input_shape corrisponda al tuo target_size

# Congelamento dei pesi del modello base
for layer in base_model.layers:
    layer.trainable = False

# Aggiunta dei nuovi strati in cima al modello
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Aggiungi un nuovo strato completamente connesso
x = Dropout(0.5)(x)  # Aggiungi un nuovo strato di dropout per prevenire l'overfitting
predictions = Dense(2, activation='softmax')(x)  # Strato di output per 2 classi

# Definizione del nuovo modello
model = Model(inputs=base_model.input, outputs=predictions)

print("Compilazione del modello")
# Compilazione del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Preparazione dei generatori di dati")
# Preparazione dei generatori di dati con preprocess_input
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Usa la funzione di preprocessamento fornita da MobileNet
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

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
    steps_per_epoch=int(train_generator.samples/train_generator.batch_size) - 1,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=int(validation_generator.samples/validation_generator.batch_size) - 1,
    callbacks=[TqdmProgressCallback()]
)

# model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=25,
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator),
#     callbacks=[TqdmProgressCallback()]
# )

print("Valutazione del modello")
# Valutazione del modello
loss, accuracy = model.evaluate(validation_generator)
print(f"Accuratezza della validazione: {accuracy}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions on the validation set
validation_predictions = model.predict(validation_generator)
predicted_classes = np.argmax(validation_predictions, axis=1)

# True classes
true_classes = validation_generator.classes

# Calculate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.savefig('confusion_matrix.png')
plt.close()

# Calculate precision and recall
# precision = precision_score(true_classes, predicted_classes, average='macro')
# recall = recall_score(true_classes, predicted_classes, average='macro')

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")

# Plot heatmap of predictions
# plt.figure(figsize=(12, 6))
# sns.heatmap(validation_predictions, cmap='viridis', xticklabels=validation_generator.class_indices.keys())
# plt.title('Prediction Heatmap')
# plt.xlabel('Class')
# plt.ylabel('Sample')
# plt.savefig('prediction_heatmap.png')
# plt.close()


print("Salvataggio del modello")
# Salvataggio del modello
model.save('yawn_detection_model_mobilenet.h5')

print("Fine dello script")
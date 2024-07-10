import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Definisci la directory del dataset con percorsi assoluti
base_dir = "/homes/greggianini/DDP"

# Verifica se la directory esiste
if not os.path.exists(base_dir):
    print(f"Directory not found: {base_dir}")
    raise FileNotFoundError(f"No such file or directory: '{base_dir}'")

# Definisci il numero di epoche e il batch size
epochs = 50
batch_size = 32

# Utilizza ImageDataGenerator per dividere i dati in training, validation e test
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(80, 80),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_data = datagen.flow_from_directory(
    base_dir,
    target_size=(80, 80),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Utilizza un altro ImageDataGenerator per creare un set di test
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

test_data = test_datagen.flow_from_directory(
    base_dir,
    target_size=(80, 80),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Modello base ResNet50V2
bmodel = ResNet50V2(include_top=False, weights='imagenet', input_tensor=Input(shape=(80, 80, 3)))
hmodel = bmodel.output

hmodel = Flatten()(hmodel)
hmodel = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(hmodel)
hmodel = BatchNormalization()(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(hmodel)
hmodel = BatchNormalization()(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2, activation='softmax')(hmodel)

model = Model(inputs=bmodel.input, outputs=hmodel)

# Sblocca più strati del modello pre-addestrato per il fine-tuning
for layer in bmodel.layers[:-20]:  # Congela più layer del modello pre-addestrato
    layer.trainable = False
for layer in bmodel.layers[-20:]:
    layer.trainable = True

model.summary()

# Definizione dei callbacks
checkpoint = ModelCheckpoint(
    os.path.join(base_dir, 'resnet50v2_model.keras'),
    monitor='val_loss', save_best_only=True, verbose=3
)

earlystop = EarlyStopping(
    monitor='val_loss', patience=10, verbose=3, restore_best_weights=True
)

learning_rate = ReduceLROnPlateau(
    monitor='val_loss', patience=3, verbose=3, factor=0.1, min_lr=1e-6
)

callbacks = [checkpoint, earlystop, learning_rate]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Utilizza tf.data.Dataset per ripetere i dati
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 80, 80, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 80, 80, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )
).repeat()

model.fit(
    train_dataset,
    steps_per_epoch=train_data.samples // batch_size,
    validation_data=validation_dataset,
    validation_steps=validation_data.samples // batch_size,
    callbacks=callbacks,
    epochs=epochs
)

acc_tr, loss_tr = model.evaluate(train_data)
print(f"Training Accuracy: {acc_tr}")
print(f"Training Loss: {loss_tr}")

acc_test, loss_test = model.evaluate(test_data)
print(f"Test Accuracy: {acc_test}")
print(f"Test Loss: {loss_test}")

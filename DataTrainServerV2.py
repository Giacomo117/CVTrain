#import zipfile
#from google.colab import drive
import print
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model

# Mount Google Drive
#drive.mount('/content/drive/')

# Definisci le directory del dataset
train_data_dir = "homes/greggianini/DDP"
train_data_dir_close = train_data_dir + "/close_eyes"
train_data_dir_open = train_data_dir + "/open_eyes"

# Definisci il numero di epoche e il batch size
epochs = 20
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_data = train_datagen.flow_from_directory(train_data_dir,
                                               target_size=(80,80),
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               subset='training')

validation_data = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(80,80),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation')

# target size means images
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(train_data_dir,
                                             target_size=(80,80),
                                             batch_size=batch_size,
                                             class_mode='categorical')

# bmodel i.e base model Inception V3
bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80,80,3)))
hmodel = bmodel.output

hmodel = Flatten()(hmodel)
hmodel = Dense(128, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2, activation='softmax')(hmodel)

model = Model(inputs=bmodel.input, outputs=hmodel)

# Sblocca gli ultimi strati del modello pre-addestrato
for layer in bmodel.layers[:-20]:
    layer.trainable = False
for layer in bmodel.layers[-20:]:
    layer.trainable = True

model.summary()

# Definizione dei callbacks
checkpoint = ModelCheckpoint('homes/greggianini/DDP/model.h5',
                             monitor='val_loss', save_best_only=True, verbose=3)

earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=3)

callbacks = [checkpoint, earlystop, learning_rate]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data,
          steps_per_epoch=train_data.samples // batch_size,
          validation_data=validation_data,
          validation_steps=validation_data.samples // batch_size,
          callbacks=callbacks,
          epochs=epochs)

acc_tr, loss_tr = model.evaluate(train_data)
print(acc_tr)
print(loss_tr)

acc_test, loss_test = model.evaluate(test_data)
print(acc_test)
print(loss_test)

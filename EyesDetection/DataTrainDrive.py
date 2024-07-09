import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, vgg16, vgg19 # For Transfer Learing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Data Augmentation

# Checking the status of GPY
tf.test.is_gpu_available()
tf.test.gpu_device_name()

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# # open -> 40,000 images
# # close -> 40,000 images
# test -> 2002 images i.e. 5% of data
# Train -> 80 % data for training and 20 % data for validation


from google.colab import drive
drive.mount('/content/drive')

batch_size = 8
# Data Augumentation for Train images means for one image in train set it will generate more 5 images of it.

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range = 0.2,
                                   shear_range=0.2,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   validation_split = 0.2)

train_data = train_datagen.flow_from_directory(r'/content/drive/MyDrive/dataFinal/train',
                                              target_size=(80,80),
                                              batch_size = batch_size,
                                              class_mode = 'categorical',
                                              subset = 'training')

validation_data = train_datagen.flow_from_directory(r'/content/drive/MyDrive/dataFinal/train',
                                              target_size=(80,80),
                                              batch_size = batch_size,
                                              class_mode = 'categorical',
                                              subset = 'validation')

# target size means images

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(r'/content/drive/MyDrive/dataFinal/train',
                                            target_size = (80,80),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

# bmodel i.e base model Inception V3
bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80,80,3)))
hmodel = bmodel.output

hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2,activation= 'softmax')(hmodel)

model = Model(inputs=bmodel.input, outputs= hmodel)
for layer in bmodel.layers:
    layer.trainable = False


model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
# checkpoint of folder , verbose progress bar
checkpoint = ModelCheckpoint(r'/content/drive/MyDrive/dataFinal/models/model.h5',
                            monitor='val_loss',save_best_only=True,verbose=3)

earlystop = EarlyStopping(monitor = 'val_loss', patience=7, verbose= 3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

callbacks=[checkpoint,earlystop,learning_rate]

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data,
          steps_per_epoch=train_data.samples//batch_size,
          validation_data=validation_data,
          validation_steps=validation_data.samples//batch_size,
          callbacks=callbacks,
          epochs=5)

acc_tr, loss_tr = model.evaluate(train_data)
print(acc_tr)
print(loss_tr)

acc_test, loss_test = model.evaluate(test_data)
print(acc_tr)
print(loss_tr)
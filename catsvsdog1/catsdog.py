import numpy as np
import pandas as pd 
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import random
import os

#build Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

#Callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
print(os.listdir("F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1"))

#Define Conastans
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#Prepare Training Data
filenames = os.listdir("F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
print ("head\n",df.head())
print ("tail\n",df.tail())
print ("info\n",df.info())
print ("describe\n",df.describe())

#see Total In count
df['category'].value_counts().plot.bar()
plt.show()

#See sample image
sample = random.choice(filenames)
print ("sample\n",sample)
print (type(sample))
#image = pil_image.open("F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1/train"+sample)

image=cv2.imread(os.path.join("F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1/train",sample))
plt.imshow(image)
plt.show()


#Build Model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(patience=10)


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

print ("callbacks\n",callbacks)

#Prepare Data
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

print (df.head())
print (df.tail())

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

train_df['category'].value_counts().plot.bar()
plt.show()

validate_df['category'].value_counts().plot.bar()
plt.show()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=2

print ("total_train\n",total_train)
print ("total_validate\n",total_validate)



#Traning Generator
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

print ("train_generator\n",train_generator)


#Validation Generator

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
print ("validation_generator\n",validation_generator)

#See how our generator work
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/castvsdog1/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
print ("example_generator\n",example_generator)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

#Fit Model
epochs=3 if FAST_RUN else 1
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
print ("coucou")


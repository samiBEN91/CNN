#https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import os
import pickle

print(os.listdir("F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/catsvsdog"))
	#print ("hello")

main_dir = "F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/catsvsdog"
train_dir = "F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/catsvsdog/train"
path = os.path.join(main_dir,train_dir)

print ("path=\n",path)	
print ("hello")

list1=[0,0,0,0,1,0,1,0,1,1]

print("sum list1=\n",sum(list1))


#print ("listdirpath=\n",os.listdir(path))

"""for p in os.listdir(path):
	print("p=\n",p)
	category = (p).split(".")[0]
	print("category=\n",category)
	convert = lambda category : int(category == 'dog')
	category = convert(category)
	print("category=\n",category)
	img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
	print("img_array=\n",img_array)
	plt.imshow(img_array)
	plt.show()
	new_img_array = cv2.resize(img_array, dsize=(80, 80))
	print("new_img_array=\n",new_img_array)
	plt.imshow(new_img_array,cmap="gray")
	plt.show()
	break"""

X = []
print("X_initialisee=\n",X)
y = []
convert = lambda category : int(category == 'dog')
def create_test_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)
        #print(category)

create_test_data(path)
#print ("category_final\n",y)
print ("y[0]\n",y[0])


#create_test_data(path)
#convert X and y into numpy array 
#We also have to reshape X with the below code
X = np.array(X).reshape(-1, 80,80,1)
#print("X avant Normalize=\n",X[120])
y = np.array(y)



#If you want to save your processed training (X) and target (y) 
#you can use pickle. Please refer the below code for this. 
#I wrote this to experiment but its not really needed. 
#But anyways I still think its better to learn. :)
#pickle.dump( X, open( "train_x", "wb" ))
#pickle.dump( y, open( "train_y", "wb" )) 

#Normalize data
X = X/255.0
#print("X apres Normalize\n",X[120])


# creer le model CNN
model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=1, batch_size=40, validation_split=0.7)


#test
test_dir = "F:/backup_these_sami/windows_10_ENSMA/Projets Logiciels Codes/intelligence_artificielle/machine learning/CNN/catsvsdog/test1"
path_test = os.path.join(main_dir,test_dir)
#os.listdir(path)

X_test = []
id_line = []
def create_test1_data(path_test):
    for ptest in os.listdir(path_test):
        id_line.append(ptest.split(".")[0])
        img_array = cv2.imread(os.path.join(path_test,ptest),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)

create_test1_data(path_test)
X_test = np.array(X_test).reshape(-1,80,80,1)
X_test = X_test/255.0

print ("id_line[5]\n",id_line[5])

predictions = model.predict(X_test)
print("predictions\n",predictions)

sumpredictions=sum(predictions)
print ("sum predictions=\n",sumpredictions)

"""predicted_val = [int(round(ptest[0])) for ptest in predictions]
print("predicted_val\n",predicted_val)

submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})
print ("top de submission_df=\n",submission_df.tail())

submission_df.to_csv("submission1.csv", index=False)"""
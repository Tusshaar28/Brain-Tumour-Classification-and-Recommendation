#
import os

from keras.src.layers import BatchNormalization

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.preprocessing.image import ImageDataGenerator as im


#Augmenting the Data
traindata = im(rescale=1./255,
               zoom_range=0.25,
              validation_split=0.15)


testdata = im(rescale=1./255)



#Splitting the Augmented Data
trainsplit = traindata.flow_from_directory("Training/", target_size=(128,128),seed=123,batch_size=16,subset="training")
validsplit = traindata.flow_from_directory("Training",target_size=(128,128),seed=123,batch_size=16,subset="validation")
testsplit = testdata.flow_from_directory("Testing",target_size=(128,128),seed=123,batch_size=16,shuffle=False)

c = trainsplit.class_indices
classes = list(c.keys())

#Importing CNN Modules
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam

#Creating the CNN
model = Sequential()

#First Convulation Layer and Pooling
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu',input_shape=(128,128,3)))
model.add(MaxPool2D((2,2)))

#Second Convulation Layer and Pooling
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))

#Dropout 20%
model.add(Dropout(0.2))

#Flattening the data for classification
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(4,activation='softmax'))

#print(model.summary())
model.compile(loss="categorical_crossentropy",optimizer = "Adam",metrics=["accuracy"])

#Fitting the Model
mn = model.fit(trainsplit, validation_data=validsplit, epochs=12, batch_size=16, verbose=1)

model.evaluate(testsplit)
model.save("braintumourclassifier3rd.h5")




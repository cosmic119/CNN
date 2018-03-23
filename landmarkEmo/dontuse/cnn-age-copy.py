import os.path
import cv2
import glob
import random
import math
import numpy as np
import keras
import dlib
import itertools
from keras.utils import np_utils
from sklearn.svm import SVC
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.models import Sequential
import time

emotions = ["anger", "sadness", "happy", "neutral", "surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-5, verbose = False)
data = {}
t1=time.time()
"""
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters=96, kernel_size=(7, 7), input_shape = (256, 256, 1), padding='same',activation = 'relu',strides=(4,4)))

# Step 2 - Pooling and LRN
classifier.add(MaxPooling2D(pool_size = (3, 3),strides=(2,2)))
classifier.add(BatchNormalization())

# Adding a second convolutional layer
classifier.add(Convolution2D(filters=256, kernel_size=(5, 5), padding='same', activation = 'relu'))

#  Pooling and LRN
classifier.add(MaxPooling2D(pool_size = (3, 3),strides=(2,2)))
classifier.add(BatchNormalization())

# Adding a third convolutional layer
classifier.add(Convolution2D(filters=384, kernel_size=(3, 3), padding='same', activation = 'relu'))

# Pooling 
classifier.add(MaxPooling2D(pool_size = (3, 3),strides=(2,2)))


# Step 3 - Flattening ??
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 8, activation = 'softmax',name='predictions'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"""
def model_generate():
    img_rows, img_cols = 16, 17
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
      
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
     
      
    model.add(Dense(7))
      
      
    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()
    return model

def get_files(emotion):
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = files[-int(len(files)*0.2):]
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(0,68): #landmark is 0~67
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist) #average x
        ymean = np.mean(ylist) #average y
        xcentral = [(x-xmean) for x in xlist] #dist x
        ycentral = [(y-ymean) for y in ylist] #dist y

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)

            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
#landmarks_vectorised[1] = x, [2] = y, [3] = dist, [4] = angle
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1: 
        data['landmarks_vestorised'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])
                training_labels.append(emotions.index(emotion))
   
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

training_data, training_labels, prediction_data, prediction_labels = make_sets()
print(training_data.shape)
training_data = np.asarray(training_data)
print(training_data.shape)
training_data = training_data.reshape(training_data.shape[0],16,17)
print("hi")
print(training_data.shape)
prediction_data = np.asarray(prediction_data)
prediction_data = prediction_data.reshape(prediction_data.shape[0],16,17)

training_data = training_data.reshape(training_data.shape[0], 16, 17,1)
prediction_data = prediction_data.reshape(prediction_data.shape[0], 16, 17,1)

training_data = np.array(training_data,dtype=np.float32)
prediction_data = np.array(prediction_data, dtype=np.float32)

training_labels = np_utils.to_categorical(training_labels, 5)
prediction_labels = np_utils.to_categorical(prediction_labels, 5)

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
datagen.fit(training_data)
model = model_generate()
filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

model.fit_generator(datagen.flow(training_data, training_labels, batch_size =10), samples_per_epoch=training_data.shape[0],validation_data = (prediction_data, prediction_labels),callbacks=[checkpointer])
"""train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow(training_data,training_labels),
                                                 target_size = (256, 256),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(prediction_data,prediction_labels,
                                            target_size = (256, 256),
                                            batch_size = 10,
                                            class_mode = 'categorical')
"""
t2=time.time()

print 'fit generator time = ', t2-t1
t1 = time.time()
model.predict_generator(generator=test_set, steps=430, verbose=1,workers=200)
t2 = time.time()
print 'predict generator time = ', t2-t1


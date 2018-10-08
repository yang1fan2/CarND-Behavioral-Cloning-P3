import csv
import cv2
import numpy as np
from scipy import ndimage
import random
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def load_image(filename, measure, correction):
    image = ndimage.imread(filename)
    return [image, np.fliplr(image)], [measure + correction, -measure - correction]

def generator(samples, batch_size =32, dir_path='data'):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                measure = float(batch_sample[3])
                # center
                pair = load_image(dir_path + "/" + batch_sample[0].strip(), measure, 0)
                images += pair[0]
                angles += pair[1]
                # left
                pair = load_image(dir_path + "/" + batch_sample[1].strip(), measure, 0.2)
                images += pair[0]
                angles += pair[1]
                #right
                pair = load_image(dir_path + "/" + batch_sample[2].strip(), measure, -0.2)
                images += pair[0]
                angles += pair[1]

            if len(images)==0:
                continue
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
lines = []
dir_path = 'data1'
def remove_leading_path(path):
    pos = path.find('IMG')
    return path[pos:]

with open(dir_path + "/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        if i==0:
            continue
        lines.append([remove_leading_path(line[0]),
                      remove_leading_path(line[1]),
                      remove_leading_path(line[2]),
                      line[3]])
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))
print(train_samples[0])
batch_size =32
version = 1
train_generator = generator(train_samples, batch_size, dir_path)
validation_generator = generator(validation_samples, batch_size, dir_path)

model =Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(3,5,5,activation='relu',subsample=(2,2)))
model.add(Convolution2D(24,5,5,activation='relu',subsample=(2,2)))
model.add(Convolution2D(36,5,5,activation='relu',subsample=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
                    train_generator, 
                    samples_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)/batch_size, nb_epoch=10, verbose = 1)
model.save('model%d.h5' % version)
print(history_object.history.keys())
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss%d.png'% version)

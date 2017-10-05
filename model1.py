import os
import csv
import cv2
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

samples = samples[1:]

# resize image to half
def resize(image):
    dim = (0.5*image.shape[0], 0.5*image.shape[1])
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# define a generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # generator never stop\
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]

            images = []
            steering_angles = []

            for batch_sample in batch_samples:
                center_source_path = batch_sample[0]
                center_filename = center_source_path.split('/')[-1]
                left_source_path = batch_sample[1]
                left_filename = left_source_path.split('/')[-1]
                right_source_path = batch_sample[2]
                right_filename = right_source_path.split('/')[-1]
                
                center_current_path = './data/IMG/' + center_filename    
                left_current_path = './data/IMG/' + left_filename
                right_current_path = './data/IMG/' + right_filename
                #print(current_path)
                center_image = cv2.imread(center_current_path)
                left_image = cv2.imread(left_current_path)
                right_image = cv2.imread(right_current_path)

                correction = 0.25
                center_steering_angle = float(batch_sample[3])
                left_steering_angle = center_steering_angle + correction
                right_steering_angle = center_steering_angle - correction

                #images.extend(center_image, left_image, right_image)
                #steering_angles.extend(center_steering_angle, left_steering_angle, right_steering_angle)
                #images.append(resize(center_image))
                #images.append(resize(left_image))
                #images.append(resize(right_image))
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                steering_angles.append(center_steering_angle)
                steering_angles.append(left_steering_angle)
                steering_angles.append(right_steering_angle)
            #x_train = np.array(images)
            #y_train = np.array(steering_angles)
            #yield shuffle(x_train, y_train)
            #yield x_train, y_train

            augmented_images, augmented_steering_angles = [], []
            #add fliped image into training set 

            for image, steering_angle in zip(images, steering_angles):
                augmented_images.append(image)
                augmented_steering_angles.append(steering_angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_steering_angles.append(-1.0*steering_angle)

            x_train = np.array(augmented_images)
            y_train = np.array(augmented_steering_angles)
            yield shuffle(x_train, y_train)


# using 3 camera (optional)

#complie and train model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#input_shape = (160, 320, 3)
# CNN structure
model = Sequential()
# parameter setup
cropping_up = 70
cropping_down = 25
# set up lambda layer
#model.add(Lambda(lambda x: (x/255.0 - 0.5, input_shape = input_shape)))
model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(160, 320, 3)))
# add cropping layer
model.add(Cropping2D(cropping=((cropping_up, cropping_down), (0, 0))))
# add convolution layers
model.add(Conv2D(24,(5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.9))
model.add(Conv2D(36,(5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.8))
model.add(Conv2D(48,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Dropout(0.8))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse',optimizer = 'adam')
history_object = model.fit_generator(train_generator, steps_per_epoch =
    len(train_samples), validation_data = 
    validation_generator, 
    validation_steps = len(validation_samples),
    epochs=5, verbose=1)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5_3')
print("model saved")

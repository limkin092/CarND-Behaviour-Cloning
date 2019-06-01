import csv
import cv2
import glob
import numpy as np

images = []
measurements = []
correction = 0.2

#path = './data/*x'
path = './data'

lines = []

for filepath in glob.glob(path):
    print(glob.glob(path))
    
    print("Scanning files in "+filepath)
    with open(filepath+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        
        #skipping first line (header line)
        next(reader)
        for line in reader:
            lines.append(line)
            
'''   
print("Augmenting images and measurements")
augmented_images, augmented_measurements = [],[]
print("Augmenting "+str(len(images))+" images")
print("Augmenting "+str(len(measurements))+" measurements")

# Flipping images and measurements to ensure more balanced dataset
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(-measurement)
'''

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import random_shift

# Splitting records to training and validation set in ration 0.2
#print(lines)

shuffledSet = shuffle(lines)
train_samples, validation_samples = train_test_split(shuffledSet, test_size=0.2)

def crop(image):
    return image[60:-25, :, :] # remove the sky and the car front

def resize(image):
    return cv2.resize(image, (200, 66), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)   

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 200 * np.random.rand(), 0
    x2, y2 = 200 * np.random.rand(), 66
    xm, ym = np.mgrid[0:66, 0:200]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  
def generate_batch(samples, batch_size=32,is_training=True):
    while 1: # Loop forever so the generator never terminates
        num_samples = len(samples)
        shuffle(samples)
        
        # Cluster of batch_size
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            # batch_sample is one line of the csv file
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/'+filename
                    image = cv2.imread(current_path)
                    #originalImage = cv2.imread(current_path)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    measurement = float(line[3])
                    
                    #Left camera 
                    if(i == 1):
                        measurement += correction       
                     #Right camera
                    if(i == 2):
                        measurement -= correction
                        
                    if(is_training):
                        image = random_shadow(image)
                        image = random_brightness(image)
                        
                    cropped_image = crop(image)
                    resized_image = resize(cropped_image)
                    yuv_image = rgb2yuv(resized_image)
                        
                    images.append(yuv_image)
                    measurements.append(measurement)
                    # Flipping
                    images.append(cv2.flip(yuv_image,1))
                    measurements.append(measurement*-1.0)
                    
            # trim image to only see section with road
            X_train = np.array(images)

            #print(len(X_train))
            y_train = np.array(measurements)
            #print(len(y_train))
            yield shuffle(X_train, y_train)
            
            
train_generator = generate_batch(train_samples,is_training = True)
validation_generator = generate_batch(validation_samples, is_training = False)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import random_shift
from math import ceil

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))
model.add(Conv2D(24, (5, 5), activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

model.compile(loss = 'mse', optimizer = 'adam')

batch_size = 32

model.fit_generator(train_generator
                    ,steps_per_epoch=ceil(len(train_samples)/batch_size)
                    ,validation_data=validation_generator
                    ,validation_steps=ceil(len(validation_samples)/batch_size)
                    ,epochs=10
                    ,verbose=1)

#model.save('model.h5')
exit()     
    
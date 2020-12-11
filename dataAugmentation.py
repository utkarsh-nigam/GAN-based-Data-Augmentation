import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import os
os.system("sudo pip install scikit-image")


x_train, y_train = np.load("Data/x_trainSet1.npy", allow_pickle=True),np.load("Data/y_trainSet1.npy", allow_pickle=True)
# print(len(x_train),len(y_train))
# print("Train:\n")
# unique, counts = np.unique(y_train, return_counts=True)
# print(dict(zip(unique, counts)))
# xTemp1=[]
# yTemp1=[]
# xTemp2=[]
# yTemp2=[]
# for i in range(len(x_train)):
#     label=y_train[i]
#     if label==0:
#         xTemp1.append(x_train[i])
#         yTemp1.append(label)
#     else:
#         xTemp2.append(x_train[i])
#         yTemp2.append(label)
#
# x_train, x_test, y_train, y_test = train_test_split(xTemp1, yTemp1, random_state=10, test_size=0.75, stratify=yTemp1)
# x_train+=xTemp2
# y_train+=yTemp2
# print(len(x_train),len(y_train))
x=[]
y=[]
for i in range(len(x_train)):
    currentImage=x_train[i]
    label=y_train[i]
    x.append(currentImage)
    y.append(label)
    # if (label==0):
    # for i in range(4):
    #     x.append(currentImage)
    #     y.append(label)
        # # flip vertically
        # x.append(np.fliplr(currentImage))
        # y.append(label)
        #
        # # flip horizontally
        # x.append(np.flipud(currentImage))
        # y.append(label)
    if (label==1):
        # flip vertically
        # x.append(np.fliplr(currentImage))
        # y.append(label)
        #
        # # flip horizontally
        # x.append(np.flipud(currentImage))
        # y.append(label)

        # horizontal shift
        samples = expand_dims(currentImage, 0)
        data_generator = ImageDataGenerator(channel_shift_range=150.0)
        it = data_generator.flow(samples, batch_size=1)
        for i in range(2):
            batch = it.next()
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)

        # horizontal shift
        samples = expand_dims(currentImage, 0)
        datagen = ImageDataGenerator(width_shift_range=0.05)
        it = datagen.flow(samples, batch_size=1)
        for i in range(2):
            batch = it.next()
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)

        # rotation
        samples = expand_dims(currentImage, 0)
        datagen = ImageDataGenerator(rotation_range=360)
        it = datagen.flow(samples, batch_size=1)
        check = []
        count = 0
        for i in range(1):
            batch = it.next()
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)


    # elif(label==2):
    #     # flip vertically
    #     x.append(np.fliplr(currentImage))
    #     y.append(label)
    #
    #     # flip horizontally
    #     x.append(np.flipud(currentImage))
    #     y.append(label)
    #
    #     # horizontal shift
    #     samples = expand_dims(currentImage, 0)
    #     datagen = ImageDataGenerator(height_shift_range=0.05)
    #     it = datagen.flow(samples, batch_size=1)
    #     for i in range(4):
    #         batch = it.next()
    #         image = batch[0].astype('uint8')
    #         x.append(image)
    #         y.append(label)
    #
    #     # rotation
    #     samples = expand_dims(currentImage, 0)
    #     datagen = ImageDataGenerator(rotation_range=360)
    #     it = datagen.flow(samples, batch_size=1)
    #     check = []
    #     count = 0
    #     for i in range(3):
    #         batch = it.next()
    #         image = batch[0].astype('uint8')
    #         x.append(image)
    #         y.append(label)
    #
    #
    # elif(label==3):
    #     # flip vertically
    #     x.append(np.fliplr(currentImage))
    #     y.append(label)
    #
    #     # flip horizontally
    #     x.append(np.flipud(currentImage))
    #     y.append(label)
    #
    #     # horizontal shift
    #     samples = expand_dims(currentImage, 0)
    #     datagen = ImageDataGenerator(height_shift_range=0.05)
    #     it = datagen.flow(samples, batch_size=1)
    #     for i in range(2):
    #         batch = it.next()
    #         image = batch[0].astype('uint8')
    #         x.append(image)
    #         y.append(label)
    #
    #     # rotation
    #     samples = expand_dims(currentImage, 0)
    #     datagen = ImageDataGenerator(rotation_range=360)
    #     it = datagen.flow(samples, batch_size=1)
    #     check = []
    #     count = 0
    #     for i in range(1):
    #         batch = it.next()
    #         image = batch[0].astype('uint8')
    #         x.append(image)
    #         y.append(label)

x, y = np.array(x), np.array(y)
x, y = shuffle(x, y)
print(x.shape, y.shape)
np.save("Data/x_trainSet3.npy", x); np.save("Data/y_trainSet3.npy", y)
print("Train:\n")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
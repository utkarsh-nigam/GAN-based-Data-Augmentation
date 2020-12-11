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

from skimage.transform import rotate, AffineTransform, warp
import skimage

# # if "train" not in os.listdir():
# os.system("unzip Data/archive.zip")
SEED = 42
testSize=0.2
valSize=0.12
# breeds=['Balinese', 'Birman', 'Sphynx - Hairless Cat','Domestic Short Hair', 'Persian', 'Domestic Long Hair', 'American Shorthair', 'Domestic Medium Hair', 'Calico', 'Dilute Calico', 'Dilute Tortoiseshell', 'Siamese', 'Ragdoll', 'Torbie', 'Tuxedo', 'Manx', 'Bengal', 'Tabby', 'Russian Blue', 'Tortoiseshell', 'Bombay', 'Snowshoe', 'Tiger', 'Maine Coon', 'Himalayan', 'Extra-Toes Cat - Hemingway Polydactyl', 'American Bobtail', 'Turkish Van', 'Turkish Angora', 'Norwegian Forest Cat', 'British Shorthair', 'Oriental Short Hair', 'Exotic Shorthair', 'Scottish Fold', 'Burmese', 'Egyptian Mau', 'Tonkinese']
breeds=["American Shorthair","Himalayan"]
x, y = [], []
# shape = {}
finalSize=64
for breedName in breeds:
    print("Breed: ",breedName)
    # shape = {}
    DATA_DIR = os.getcwd() + "/images/"+breedName+"/"+breedName+"/"
    for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
        currImage=cv2.imread(DATA_DIR + path)
        currShape=currImage.shape
        if currShape == (400, 300, 3):
            desired_size = 400
            old_size = currImage.shape[:2]  # old_size is in (height, width) format
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = cv2.resize(currImage, (new_size[1], new_size[0]))
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)
            finalImage = cv2.resize(new_im, (finalSize, finalSize))
            x.append(finalImage)
            y.append(breedName)

    # print(shape)

x, y = np.array(x), np.array(y)
le = LabelEncoder()
le.fit(breeds)
y = le.transform(y)
# print(shape)
print(x.shape, y.shape)
# for i in range(0,5):
print(x[0])
print(y[0])
x, y = shuffle(x, y)

# X_data,Y_data=np.load("Data/x_trainSet1.npy", allow_pickle=True), np.load("Data/y_trainSet1.npy", allow_pickle=True)
# print(Y_data)
# Y_data= to_categorical(Y_data, num_classes=4)
# print(Y_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=testSize, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=SEED, test_size=valSize, stratify=y_train)


np.save("Data/x_trainSet1.npy", x_train);
np.save("Data/y_trainSet1.npy", y_train);
np.save("Data/x_val.npy", x_val);
np.save("Data/y_val.npy", y_val);
np.save("Data/x_test.npy", x_test);
np.save("Data/y_test.npy", y_test);

'''

# DATA_DIR=os.getcwd()+"/trainNew/"
# RESIZE_TO=50


# print(x_temp[0])
# print(y_temp[0])
# x=x_temp[0]
count=1
xTemp=np.load("x_train.npy")
x=xTemp[1]
plt.figure(figsize=(8, 8))
plt.subplot(3,5,count)
count+=1
plt.imshow(x)
plt.title("Actual Photo")

plt.subplot(3,5,count)
count+=1
plt.imshow(np.fliplr(x))
plt.title("Horizontal Flip")

plt.subplot(3,5,count)
count+=1
plt.imshow(np.flipud(x))
plt.title("Vertical Flip")
# plt.show()


# vertical shift
samples = expand_dims(x, 0)
datagen = ImageDataGenerator(width_shift_range=0.1)
it = datagen.flow(samples, batch_size=1)

for i in range(3):
    plt.subplot(3, 5, count)
    count += 1
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.title("Horizontal Shift:" + str(i + 1))
# plt.show()
# horizontal shift
samples = expand_dims(x, 0)
datagen = ImageDataGenerator(height_shift_range=0.1)
it = datagen.flow(samples, batch_size=1)
for i in range(3):
    plt.subplot(3, 5, count)
    count += 1
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.title("Vertical Shift:" + str(i + 1))
# plt.show()

samples = expand_dims(x, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=360)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# plt.figure()
# count=0
for i in range(6):
    plt.subplot(3, 5, count)
    count += 1
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.title("Rotation:"+str(i+1))
plt.show()





# #
# #
# #
# # samples = expand_dims(x, 0)
# # # create image data augmentation generator
# # datagen = ImageDataGenerator(height_shift_range=0.05)
# # # prepare iterator
# # it = datagen.flow(samples, batch_size=1)
# # # generate samples and plot
# # check=[]
# #
# # for i in range(9):
# #     plt.subplot(3, 3, i + 1)
# #     batch = it.next()
# #     image = batch[0].astype('uint8')
# #     comparison = x == image
# #     equal_arrays = comparison.all()
# #     if(equal_arrays==True):
# #         print("Same")
# #     plt.imshow(image)
# #     plt.title(str(i+1))
# #     check.append(image)
# # plt.show()
# #
# #
# # for i in range(0,len(check)):
# #     for j in range(0,len(check)):
# #         if (i!=j):
# #             comparison = check[i] == check[j]
# #             equal_arrays = comparison.all()
# #             # print(equal_arrays)
# #             if (equal_arrays == True):
# #                 print(i+1,j+1)
# #                 print("Same Linear Height")
# #
# #
# # samples = expand_dims(x, 0)
# # # create image data augmentation generator
# # datagen = ImageDataGenerator(width_shift_range=0.05)
# # # prepare iterator
# # it = datagen.flow(samples, batch_size=1)
# # # generate samples and plot
# #
# # check=[]
# #
# # for i in range(9):
# #     plt.subplot(3, 3, i + 1)
# #     batch = it.next()
# #     image = batch[0].astype('uint8')
# #     comparison = x == image
# #     equal_arrays = comparison.all()
# #     if(equal_arrays==True):
# #         print("Same")
# #     plt.imshow(image)
# #     plt.title(str(i+1))
# #     check.append(image)
# # plt.show()
# #
# #
# # for i in range(0,len(check)):
# #     for j in range(0,len(check)):
# #         if (i!=j):
# #             comparison = check[i] == check[j]
# #             equal_arrays = comparison.all()
# #             print(equal_arrays)
# #             if (equal_arrays == True):
# #                 print(i+1,j+1)
# #                 print("Same Linear Width")
# #
# #
# # samples = expand_dims(x, 0)
# # # create image data augmentation generator
# # datagen = ImageDataGenerator(rotation_range=360)
# # # prepare iterator
# # it = datagen.flow(samples, batch_size=1)
# # # generate samples and plot
# # check=[]
# # count=0
# # for i in range(32):
# #     plt.subplot(3, 3, count + 1)
# #     batch = it.next()
# #     image = batch[0].astype('uint8')
# #     comparison = x == image
# #     equal_arrays = comparison.all()
# #     if(equal_arrays==True):
# #         print("Same")
# #     plt.imshow(image)
# #     plt.title(str(i+1))
# #     check.append(image)
# #     if count==8:
# #         count=0
# #         plt.show()
# #     else:
# #         count+=1
# # plt.show()
# #
# # for i in range(0,len(check)):
# #     for j in range(0,len(check)):
# #         if (i!=j):
# #             comparison = check[i] == check[j]
# #             equal_arrays = comparison.all()
# #             print(equal_arrays)
# #             if (equal_arrays == True):
# #                 print(i+1,j+1)
# #                 print("Same Rotation")
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# np.save("X_TrainDataFinal.npy", x_train);
# np.save("X_ValDataFinal.npy", x_val);
# np.save("X_TestDataFinal.npy", x_test);
# np.save("Y_TrainDataFinal.npy", y_train);
# np.save("Y_ValDataFinal.npy", y_val);
# np.save("Y_TestDataFinal.npy", y_test);

# filearray=[["X_TrainDataFinal.npy","Y_TrainDataFinal.npy"],["X_ValDataFinal.npy","Y_ValDataFinal.npy"],["X_TestDataFinal.npy","Y_TestDataFinal.npy"]]
#
# for j in range(len(filearray)):
#     currentArray=filearray[j]
#     print("Current Files: ",currentArray)
#     xfilename=currentArray[0]
#     yfilename=currentArray[1]
#     x_temp, y_temp = np.load(xfilename), np.load(yfilename)
#
#     x, y = [], []
#     for i in range(len(y_temp)):
#         # print(path[:-4])
#         currentImage=x_temp[i]
#         x.append(currentImage)
#         label = y_temp[i]
#         y.append(label)
#         if (label==1):
#             # flip vertically
#             x.append(np.fliplr(currentImage))
#             y.append(label)
#
#             # flip horizontally
#             x.append(np.flipud(currentImage))
#             y.append(label)
#
#             # vertical shift
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(width_shift_range=0.05)
#             it = datagen.flow(samples, batch_size=1)
#             for i in range(6):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#             # horizontal shift
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(height_shift_range=0.05)
#             it = datagen.flow(samples, batch_size=1)
#             for i in range(6):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#             # rotation
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(rotation_range=360)
#             it = datagen.flow(samples, batch_size=1)
#             check = []
#             count = 0
#             for i in range(6):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#
#         elif(label==2):
#             # flip vertically
#             x.append(np.fliplr(currentImage))
#             y.append(label)
#
#             # flip horizontally
#             x.append(np.flipud(currentImage))
#             y.append(label)
#
#             # vertical shift
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(width_shift_range=0.05)
#             it = datagen.flow(samples, batch_size=1)
#             for i in range(9):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#             # horizontal shift
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(height_shift_range=0.05)
#             it = datagen.flow(samples, batch_size=1)
#             for i in range(9):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#             # rotation
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(rotation_range=360)
#             it = datagen.flow(samples, batch_size=1)
#             check = []
#             count = 0
#             for i in range(32):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#
#         elif(label==3):
#             # flip vertically
#             x.append(np.fliplr(currentImage))
#             y.append(label)
#
#             # flip horizontally
#             x.append(np.flipud(currentImage))
#             y.append(label)
#
#             # rotation
#             samples = expand_dims(currentImage, 0)
#             datagen = ImageDataGenerator(rotation_range=360)
#             it = datagen.flow(samples, batch_size=1)
#             check = []
#             count = 0
#             for i in range(4):
#                 batch = it.next()
#                 image = batch[0].astype('uint8')
#                 x.append(image)
#                 y.append(label)
#
#
#
#
#     x, y = np.array(x), np.array(y)
#     print(x.shape, y.shape)
#     np.save(xfilename, x); np.save(yfilename, y)
#
#     # print(xTest/255)
#
#
#     # a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
#     unique, counts = np.unique(y, return_counts=True)
#     print(dict(zip(unique, counts)))'''
import numpy as np
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, Concatenate,  Flatten
from keras.initializers import glorot_uniform, glorot_normal
from keras.optimizers import Adam, SGD
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

breedName = "Himalayan"
finalSize=64
x_train=[]
y_train=[]
DATA_DIR= os.getcwd() + "/finalImages/Train/"+breedName+"/"+breedName+"/"
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    currImage = cv2.imread(DATA_DIR + path)
    currImage = cv2.resize(currImage, (finalSize, finalSize))
    y_train.append(breedName)
    x_train.append(currImage)
y_train[0]="American Shorthair"
y_train[1]="Himalayan"
y_train[2]="Russian Blue"
x, y = np.array(x_train), np.array(y_train)
le = LabelEncoder()
breeds=["American Shorthair","Himalayan", "Russian Blue", "Tabby"]
le.fit(breeds)
y = le.transform(y)



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
weight_init = glorot_normal(seed=SEED)


X_original = x.copy()
X_real = np.ndarray(shape=(X_original.shape[0], 64, 64, 3))
for i in range(X_original.shape[0]):
    X_real[i] = cv2.resize(X_original[i], (64, 64))

y_real = y.copy()
n_classes = y_real.max() - y_real.min() + 1
img_size = X_real[0].shape

# latent space of noise
z = (100,)
optimizer = Adam(lr=0.0002, beta_1=0.5)

# Build Generator
def generator_conv():
    label = Input((1,), dtype='int32')
    noise = Input(shape=z)

    le = Embedding(n_classes, 100)(label)
    le = Dense(4*4)(le)
    le = Reshape((4, 4, 1))(le)

    noi = Dense(4 * 4 * 256)(noise)
    noi = LeakyReLU(alpha=0.2)(noi)
    noi = Reshape((4, 4, 256))(noi)
    
    merge = Concatenate()([noi, le])

    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same')(merge)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)

    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)

    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    generated = Conv2D(3, (8, 8), padding='same', activation='tanh')(x)

    generator = Model(inputs=[noise, label], outputs=generated)
    return generator


def discriminator_conv():
    label = Input((1,), dtype='int32')
    img = Input(img_size)

    le = Embedding(n_classes, 100)(label)
    le = Dense(img_size[0] * img_size[1])(le)
    le = Reshape((img_size[0], img_size[1], 1))(le)

    merge = Concatenate()([img, le])
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(merge)
    x = LeakyReLU(0.2)(x)


    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    ## Size: 4 x 4 x 128
    # x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    discriminator = Model(inputs=[img, label], outputs=out)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

def generator_trainer(generator, discriminator):
    discriminator.trainable = False
    gen_noise, gen_label = generator.input
    gen_out = generator.output
    out = discriminator([gen_out, gen_label])
    model = Model([gen_noise, gen_label], out)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

# GAN model compiling
class GAN():
    def __init__(self, img_shape=(64, 64, 3), latent_space=(100,)):
        self.img_size = img_shape  # channel_last
        self.z = latent_space
        self.optimizer = Adam(0.0002, 0.5)
        self.gen = generator_conv()
        self.discr = discriminator_conv()
        self.train_gen = generator_trainer(self.gen, self.discr)
        self.loss_D, self.loss_G = [], []

    def train(self, dataset, epochs=50, batch_size=64):
        # load data
        imgs, labels = dataset
        imgs = (imgs - 127.5)/127.5
        bs_half = batch_size//2

        for epoch in range(epochs):
            # Get a half batch of random real images
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img, real_label = imgs[idx], labels[idx]
            noise = np.random.normal(0, 1, size=((bs_half,) + self.z))
            noise_label = np.random.randint(0, n_classes, bs_half)
            fake_img = self.gen.predict([noise, noise_label])
            real = np.random.uniform(0.9, 1.0, (bs_half, 1))
            fake = np.zeros((bs_half, 1))
            mixpoint = int(bs_half * 0.95)
            real = np.concatenate([real[:mixpoint], fake[mixpoint:]])
            fake = np.concatenate([fake[:mixpoint], real[mixpoint:]])
            np.random.shuffle(real)
            np.random.shuffle(fake)
            loss_fake = self.discr.train_on_batch([fake_img, noise_label], fake)
            loss_real = self.discr.train_on_batch([real_img, real_label], real)
            self.loss_D.append(0.5 * np.add(loss_fake, loss_real))
            noise = np.random.normal(0, 1, size=((batch_size,) + self.z))
            noise_label = np.random.randint(0, n_classes, batch_size)
            loss_gen = self.train_gen.train_on_batch([noise, noise_label], np.ones(batch_size))
            self.loss_G.append(loss_gen)
            if ((epoch + 1) * 10) % epochs == 0:
                print('Epoch (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, epochs, loss_real[0], loss_fake[0], 100*self.loss_D[-1][1], loss_gen))

        return


def saveGeneratedImages(gan,id):
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, 100))
    noise_label = np.ones((1, 16)).reshape(-1, 1)

    gen_imgs = gan.gen.predict([noise, noise_label])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    cnt = 0
    for j in range(c):
        plt.figure()
        plt.imshow(gen_imgs[cnt])
        plt.set_title("Cat type: "+str(breedName)
        plt.axis('off')
        plt.savefig("step = "+str(id)+".png")
        plt.show(
        cnt += 1
    return


gan = GAN()
LEARNING_STEPS = 15000
BATCH_SIZE = 32
EPOCHS = 50
for learning_step in range(LEARNING_STEPS):
    gan.train([X_real, y_real], epochs=EPOCHS, batch_size=BATCH_SIZE)
    if (learning_step + 1) % 10 == 0:
        checkPrint=(learning_step + 1)
        saveGeneratedImages(gan,checkPrint)
        gan.gen.save('himalayan_'+str(checkPrint)+'.h5')

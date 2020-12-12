# Image Augmentation Using GAN

## Objective
Dataset imbalance is one of the major problems that we face while training any machine learning model for classification. With this thought, this projects aims to
use GAN's to improve the performance of CNN models for classification of highly imbalanced image dataset.

## Dataset
Dataset being used for this project is based on Cat's Breed dataset which is sourced from Kaggle https://www.kaggle.com/ma7555/cat-breeds-dataset. It consists of 67 different breeds of cats. However, due to computational limitations, we choose 2 breeds (American Shorthair and Himalayan) that when combined together give us a highly imbalanced dataset.

![Alt text](/assets/img/img1.png?raw=true "")

![Alt text](/assets/img/img5.png?raw=true "")


## Preprocessing
This dataset had images of variable size and aspect ratio. Therefore, we decided to go with the image of size 400 X 300 (height X width). Reason for this choice, was to provide the model with images that capture the maximum features of the cats. And with this particular image size, we observed maximum images to have cats standing, thus giving maximum visibility in terms of features. Following are the steps of pre processing:

![Alt text](/assets/img/img2.png?raw=true "")
(a) Original 400 X 300-pixels image, (b) Adding padding of 50 pixels on both horizontal sides, thereby making image 400 X 400- pixel,
(c) Further, rezsized to 64 X 64-pixels


## Architectures
To evaluate the performance of GAN's, we decided to choose four of the top pre-trained models for our study:
•	AlexNet 
•	DenseNet
•	VGG19 
•	ResNext

Since our objective is to focus on improving the performance by increasing the sample of minority class, hence we decided to keep the models to their base configuration (meaning no hyper tuning was done).

## Methododlogy
To compare the results, we decided to do this experiment in three phases as follows:

### No data Augmentation
Train models on imbalanced data set

### Conventional Augmentation
To increase the samples of minorty class, we applied some conventional augmentation methods such as flipping, rotation, vertical and horizontal shifts.
![Alt text](/assets/img/img3.png?raw=true "")

### GAN Augmentation
Below is the architecture of GAN:
![Alt text](/assets/img/img4.png?raw=true "")

Generator learns to generate data plausible samples, which become negative training samples for the discriminator. Where as discriminator learns to differentiate the fake samples provided by generator from real data samples. In this process, generator is penalized by discriminator for producing implausible results. In nut shell, both the networks compete to beat each other. Discrimnitor to recognise the generated images as fake, and generator to fool the discrimnator so that it is not able to differentiate between real and fake images.

We trained GAN only for minority class i.e., Himalayan Breed. Below are the results from GAN after training for 20,000 epochs.
![Alt text](/assets/img/img6.png?raw=true "")

## Results
### Confusion Matrix
![Alt text](/assets/img/img7.png?raw=true "")

We can clearly see for the case of No Augmentation, training on imbalanced data set leads to confusion for the model, and as a result it gets biased toards the majority class i.e., American Shorthair.
However, after increasing the sample of minority class using Conventional Augmentation, does give us better results, and the model starts to recognise Himalayan breed. And, in case of GAN augmentation, we see even more improvement in terms of classification.

### Loss, F1 Score, and Cohen Kappa Score on Test Dataset
![Alt text](/assets/img/img8.png?raw=true "")

We can observe definitive improvement in all the models exceot for VGG19, which can be further investigated.

## Conclusion
Through this project, we observed how imbalanced dataset can be problematic in the training of classification models, and how it can be taken care of by using conventional and advanced methods.







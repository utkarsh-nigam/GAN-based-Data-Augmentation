# Image Augmentation Using GAN

## Objective
Dataset imbalance is one of the major problems that we face while training any machine learning model for classification. With this thought, this projects aims to
use GAN's to improve the performance of CNN models for classification of highly imbalanced image dataset.

## Dataset
Dataset being used for this project is based on Cat's Breed dataset which is sourced from Kaggle https://www.kaggle.com/ma7555/cat-breeds-dataset. It consists of 67 different breeds of cats. However, due to computational limitations, we choose 2 breeds that when combined together give us a highly imbalanced dataset.

![Alt text](/assets/img/img1.png?raw=true "")

## Preprocessing
This dataset had images of variable size and aspect ratio. Therefore, we decided to go with the image of size 400 X 300 (height X width). Reason for this choice, was to provide the model with images that capture the maximum features of the cats. And with this particular image size, we observed maximum images to have cats standing, thus giving maximum visibility in terms of features. Following are the steps of pre processing:

![Alt text](/assets/img/img2.png?raw=true "")
(a) Original 400 X 300-pixel images, (b) Adding padding of 50 pixels on both horizontal sides, thereby making image 400 X 400- pixel, (c) Further, rezsized to 64 X 64-pixel


## Architectures
To evaluate the performance of GAN's, we decided to choose four of the top pre-trained models for our study:
•	AlexNet 
•	DenseNet
•	VGG19 
•	ResNext

Since our objective is to focus on improving the performance by increasing the sample of minority class, hence we decided to keep the models to their base configuration (meaning no hyper tuning was done).

## Methododlogy
To compare the results, we decided to do this experiment in three phases as follows:

### 





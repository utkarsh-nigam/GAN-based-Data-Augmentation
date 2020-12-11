# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


DROPOUT = 0.1

datatype="GAN Augmentation"
subFolder="Day3"

# if modelType=="Densenet":
# elif modelType == "VGG19":
# elif modelType == "Resnet":
# elif modelType == "Alexnet":

modelType="VGG19"
epochNumber=3


comments="EpochNumber_"+str(epochNumber)

targetClasses=["American Shorthair","Himalayan"]

def plotConfusionMat(y,x,title="Title", return_labels=False):
    with torch.no_grad():
        # logits = model(x)

        cm=confusion_matrix(y, x)

        df_cm = pd.DataFrame(cm, columns=targetClasses, index=targetClasses)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        plt.figure()
        # sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm,cmap="Blues", annot=True, fmt='d')  # font size
        # plt.ylabel("True Labels")
        # plt.xlabel("Predicted Labels")
        plt.xticks(rotation=45)
        plt.title(title)
        plt.savefig("Results/"+title+".png")
        plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Resnext50(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnext50_32x4d()
        # for param in model.parameters():
        #     param.requires_grad = False

        resnet.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2))
        self.finalModel=resnet


    def forward(self, x):
        return self.finalModel(x)


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19_bn(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        vgg.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2))
        self.finalModel=vgg


    def forward(self, x):
        return self.finalModel(x)


class Densenet(nn.Module):
    def __init__(self):
        super().__init__()
        denseModel = models.densenet121(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        denseModel.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2))
        self.finalModel=denseModel


    def forward(self, x):
        return self.finalModel(x)


class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        alexnetModel = models.alexnet(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        alexnetModel.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2))
        self.finalModel=alexnetModel


    def forward(self, x):
        return self.finalModel(x)


x_test,y_test=np.load("Data/x_test.npy", allow_pickle=True), np.load("Data/y_test.npy", allow_pickle=True)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if modelType=="Densenet":
    BATCH_SIZE = 35
    model = Densenet().to(device)

elif modelType == "VGG19":
    BATCH_SIZE = 25
    model = VGG().to(device)

elif modelType == "Resnet":
    BATCH_SIZE = 35
    model = Resnext50().to(device)

elif modelType == "Alexnet":
    BATCH_SIZE = 35
    model = Alexnet().to(device)

criterion = nn.CrossEntropyLoss()

modelPath="Models/"+subFolder+"/Epoch_" + str(epochNumber) + "_model_"+modelType+".pt"
model.load_state_dict(torch.load(modelPath))
model.eval()


yPredTestBatch=[]
if ((len(x_test) % BATCH_SIZE) == 0):
    loopRun2 = len(x_test) // BATCH_SIZE
else:
    loopRun2 = len(x_test) // BATCH_SIZE + 1
loss_test = 0
flag2 = 1
for batch2 in range(loopRun2):
    if ((batch2 + 1) * BATCH_SIZE == len(x_test) - 1):
        inds2 = slice(batch2 * BATCH_SIZE, len(x_test) + 1)
        flag2 = 0
    else:
        inds2 = slice(batch2 * BATCH_SIZE, (batch2 + 1) * BATCH_SIZE)
    imgs2 = x_test[inds2]
    tempXtest = []
    for currentTemp in imgs2:
        tempXtest.append(preprocess(currentTemp))
    imgs2 = torch.stack(tempXtest, 0)
    imgs2, targets2 = torch.FloatTensor(imgs2),  torch.from_numpy(y_test[inds2])
    imgs2, targets2 = imgs2.to(device), targets2.to(device)
    with torch.no_grad():
        y_test_pred = model(imgs2)
        for value in y_test_pred:
            yPredTestBatch.append(value)
        loss = criterion(y_test_pred, targets2)
        loss_test += loss.item()
    if flag2 == 0:
        break

lossTestFinal=loss_test/loopRun2

yPredTestBatch = torch.stack(yPredTestBatch, 0)
with torch.no_grad():
    yPredTestBatch=np.argmax(yPredTestBatch.cpu().numpy(), axis=1)

accTestFinal=100 * accuracy_score(y_test, yPredTestBatch)
f1ScoreTestFinal=f1_score(y_test, yPredTestBatch, average='macro')
chScoreTestFinal = cohen_kappa_score(y_test, yPredTestBatch)

print(
    "Test Loss {:.5f} - Test Accuracy {:.5f} - Test f1Score {:.5f} - Test Cohen Kappa Score {:.5f}".format(lossTestFinal, accTestFinal, f1ScoreTestFinal, chScoreTestFinal))

# plotConfusionMat(yPredTestBatch, y_test)


#
# if "Results/testResults.csv" not in os.listdir():
#     column_names = ["ID", "Model Type","Data Type","Day", "Test Accuracy", "Test Loss",
#                     "Test F1 Score", "Test Cohen Kappa Score","Overall Score","Comments"]
#
#     data = pd.DataFrame(columns=column_names)
#     data.to_csv("Results/testResults.csv", index=False)

data=pd.read_csv("Results/testResults.csv")
modelID=data.shape[0]
modelID+=1
plotConfusionMat(y_test,yPredTestBatch,modelType+"_ID_"+str(modelID))
row = {"ID":modelID, "Model Type":modelType,"Data Type":datatype,"Day":subFolder, "Test Accuracy":accTestFinal, "Test Loss":lossTestFinal,
      "Test F1 Score":f1ScoreTestFinal,"Test Cohen Kappa Score":chScoreTestFinal,"Overall Score":(f1ScoreTestFinal+chScoreTestFinal)/2,"Comments":comments}

data = data.append(row, ignore_index=True)
data.to_csv("Results/testResults.csv", index=False)
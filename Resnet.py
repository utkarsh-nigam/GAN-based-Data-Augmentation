# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from torchvision import transforms
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

SEED = 42
LR = 0.001
N_EPOCHS = 15
BATCH_SIZE = 35
DROPOUT = 0.1
testSize=0.2
valSize=0.12

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datatype="GAN Augmentation"
subFolder="Day3"
modelType="Resnet"
comments=""
if "Models" not in os.listdir():
    os.mkdir("Models")
    path = "Models/" + subFolder
    os.mkdir(path)
# path="Models/"+subFolder
#
# if path not in os.listdir():
#     os.mkdir(path)

# %% ----------------------------------- Model Setup -------------------------------------------------------------------

#
# def acc(x):
#     with torch.no_grad():
#         # logits = model(x)
#         pred_labels = np.argmax(x.cpu().numpy(), axis=1)
#     if return_labels:
#         return pred_labels
#     else:
#         return 100*accuracy_score(y, pred_labels)



def plotConfusionMat(x, y, return_labels=False):
    with torch.no_grad():
        # logits = model(x)
        pred_labels = np.argmax(x.cpu().numpy(), axis=1)
        cm=confusion_matrix(y, pred_labels)
        plt.figure()
        # sns.set(font_scale=1.4)  # for label size
        sns.heatmap(cm, annot=True, annot_kws={"size": 8})  # font size
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")
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


# ,Y_data=np.load("Data/x_trainSet1.npy", allow_pickle=True), np.load("Data/y_trainSet1.npy", allow_pickle=True)
# # print(Y_data)
# # Y_data= to_categorical(Y_data, num_classes=4)
# # print(Y_data)
# x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, random_state=SEED, test_size=testSize, stratify=Y_data)
x_train, x_val, y_train, y_val = np.load("Data/x_trainSet3.npy", allow_pickle=True), np.load("Data/x_val.npy", allow_pickle=True),np.load("Data/y_trainSet3.npy", allow_pickle=True), np.load("Data/y_val.npy", allow_pickle=True)#train_test_split(x_train, y_train, random_state=SEED, test_size=valSize, stratify=y_train)
# print(y_val)
# %% -------------------------------------- Training Prep --------------------------------------------------------------

# Initialize the model
model = Resnext50().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop --------------------------------------------------------------
print("Starting training loop...")
trainPlot = []
testPlot = []
history={"TrainLoss":[],"ValLoss":[],"TrainAccuracy":[],"ValAccuracy":[],"Trainf1Score":[],"Valf1Score":[],"TrainchScore":[],"ValchScore":[]}

for epoch in range(N_EPOCHS):
    yPredTrainBatch=[]
    yPredValBatch = []
    torch.cuda.empty_cache()
    loss_train = 0
    model.train()
    flag = 1
    if ((len(x_train) % BATCH_SIZE) == 0):
        loopRun = len(x_train) // BATCH_SIZE
    else:
        loopRun = len(x_train) // BATCH_SIZE + 1
    for batch in range(loopRun):
        if ((batch + 1) * BATCH_SIZE == len(x_train) - 1):
            inds = slice(batch * BATCH_SIZE, len(x_train) + 1)
            flag = 0
        else:
            inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)

        imgs = x_train[inds]
        tempXtrain = []
        for currentTemp in imgs:
            tempXtrain.append(preprocess(currentTemp))
        imgs = torch.stack(tempXtrain, 0)
        imgs.requires_grad = True
        imgs, targets = torch.FloatTensor(imgs), torch.from_numpy(y_train[inds])
        imgs, targets = imgs.to(device), targets.to(device)
        # targets = torch.tensor(targets, dtype=torch.long, device=device)
        # print(targets)
        optimizer.zero_grad()
        logits = model(imgs)
        for value in logits:
            yPredTrainBatch.append(value)
        # print(yPredTrainBatch)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        if flag == 0:
            break

    model.eval()
    if ((len(x_val) % BATCH_SIZE) == 0):
        loopRun2 = len(x_val) // BATCH_SIZE
    else:
        loopRun2 = len(x_val) // BATCH_SIZE + 1
    loss_val = 0
    flag2 = 1
    for batch2 in range(loopRun2):
        if ((batch2 + 1) * BATCH_SIZE == len(x_val) - 1):
            inds2 = slice(batch2 * BATCH_SIZE, len(x_val) + 1)
            flag2 = 0
        else:
            inds2 = slice(batch2 * BATCH_SIZE, (batch2 + 1) * BATCH_SIZE)
        imgs2 = x_val[inds2]
        tempXtest = []
        for currentTemp in imgs2:
            tempXtest.append(preprocess(currentTemp))
        imgs2 = torch.stack(tempXtest, 0)
        imgs2, targets2 = torch.FloatTensor(imgs2), torch.from_numpy(y_val[inds2])
        imgs2, targets2 = imgs2.to(device), targets2.to(device)
        # targets2 = torch.tensor(targets2, dtype=torch.long, device=device)
        with torch.no_grad():
            y_test_pred = model(imgs2)
            for value in y_test_pred:
                yPredValBatch.append((value))
            loss = criterion(y_test_pred, targets2)
            loss_val += loss.item()
        if flag2 == 0:
            break

    lossTrainFinal=loss_train/loopRun
    lossValFinal=loss_val/loopRun2
    # print(yPredTrainBatch)
    yPredTrainBatch = torch.stack(yPredTrainBatch, 0)
    yPredValBatch = torch.stack(yPredValBatch, 0)
    with torch.no_grad():
        yPredTrainBatch=np.argmax(yPredTrainBatch.cpu().numpy(), axis=1)
        yPredValBatch = np.argmax(yPredValBatch.cpu().numpy(), axis=1)
    accTrainFinal=100 * accuracy_score(y_train, yPredTrainBatch)
    f1ScoreTrainFinal=f1_score(y_train, yPredTrainBatch, average='macro')
    chScoreTrainFinal = cohen_kappa_score(y_train, yPredTrainBatch)

    accValFinal=100 * accuracy_score(y_val, yPredValBatch)
    f1ScoreValFinal = f1_score(y_val, yPredValBatch, average='macro')
    chScoreValFinal = cohen_kappa_score(y_val, yPredValBatch)


    print("Epoch {} | Train Loss {:.5f} - Train Accuracy {:.5f} | Validation Loss {:.5f} - Validation Accuracy {:.5f}".format(
        epoch, lossTrainFinal, accTrainFinal, lossValFinal,accValFinal))

    history["TrainLoss"].append(lossTrainFinal)
    history["ValLoss"].append(lossValFinal)

    history["TrainAccuracy"].append(accTrainFinal)
    history["ValAccuracy"].append(accValFinal)

    history["Trainf1Score"].append(f1ScoreTrainFinal)
    history["Valf1Score"].append(f1ScoreValFinal)

    history["TrainchScore"].append(lossTrainFinal)
    history["ValchScore"].append(chScoreValFinal)
    torch.save(model.state_dict(), "Models/"+str(subFolder)+"/Epoch_" + str(epoch) + "_model_"+modelType+".pt")
# print(history)

# if "Results" not in os.listdir():
#     os.mkdir("Results")
#
# if "Results/compileResults.csv" not in os.listdir():
#     column_names = ["ID", "Model Type","Data Type","Day", "Train Accuracy", "Train Loss",
#                     "Train F1 Score", "Train Cohen Kappa Score","Validation Accuracy","Validation Loss",
#                     "Validation F1 Score", "Validation Cohen Kappa Score","Comments"]
#
#     data = pd.DataFrame(columns=column_names)
#     data.to_csv("Results/compileResults.csv", index=False)

data=pd.read_csv("Results/compileResults.csv")
modelID=data.shape[0]
modelID+=1

row = {"ID":modelID, "Model Type":modelType,"Data Type":datatype,"Day":subFolder, "Train Loss":history["TrainLoss"], "Train Accuracy":history["TrainAccuracy"],
                    "Train F1 Score":history["Trainf1Score"], "Train Cohen Kappa Score":history["TrainchScore"], "Validation Loss":history["ValLoss"],"Validation Accuracy":history["ValAccuracy"],
                    "Validation F1 Score":history["Valf1Score"], "Validation Cohen Kappa Score":history["ValchScore"],"Comments":comments}

data = data.append(row, ignore_index=True)
# data = data.sort_values(by=["Stacked Score"], ascending=False)
data.to_csv("Results/compileResults.csv", index=False)

#
# yPredTestBatch=[]
# model.eval()
# if ((len(x_test) % BATCH_SIZE) == 0):
#     loopRun2 = len(x_test) // BATCH_SIZE
# else:
#     loopRun2 = len(x_test) // BATCH_SIZE + 1
# loss_test = 0
# flag2 = 1
# for batch2 in range(loopRun2):
#     if ((batch2 + 1) * BATCH_SIZE == len(x_test) - 1):
#         inds2 = slice(batch2 * BATCH_SIZE, len(x_test) + 1)
#         flag2 = 0
#     else:
#         inds2 = slice(batch2 * BATCH_SIZE, (batch2 + 1) * BATCH_SIZE)
#     imgs2 = x_test[inds2]
#     tempXtest = []
#     for currentTemp in imgs2:
#         tempXtest.append(preprocess(currentTemp))
#     imgs2 = torch.stack(tempXtest, 0)
#     imgs2, targets2 = torch.FloatTensor(imgs2),  torch.from_numpy(y_test[inds2])
#     imgs2, targets2 = imgs2.to(device), targets2.to(device)
#     with torch.no_grad():
#         y_test_pred = model(imgs2)
#         for value in y_test_pred:
#             yPredTestBatch.append(value)
#         loss = criterion(y_test_pred, targets2)
#         loss_test += loss.item()
#     if flag2 == 0:
#         break
#
# lossTestFinal = loss_test / BATCH_SIZE
# yPredTestBatch=torch.stack(yPredTestBatch, 0)
# accTestFinal = acc(yPredTestBatch, y_test)
# print(
#     "Test Loss {:.5f} - Test Accuracy {:.5f} ".format(lossTestFinal, accTestFinal))
#
# plotConfusionMat(yPredTestBatch, y_test)

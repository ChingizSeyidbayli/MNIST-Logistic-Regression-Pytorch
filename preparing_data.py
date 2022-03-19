import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv(r"../train.csv",dtype = np.float32)

# split data into features (pixels) and labels (numbers from 0 to 9)
output_val = train.label.values
input_val = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%
input_val_train, input_val_test, output_val_train, output_val_test = train_test_split(input_val,
                                                                              output_val,
                                                                              test_size = 0.2,
                                                                              random_state = 42)

# convert numpy array to torch tensor
inputTrain = torch.from_numpy(input_val_train)
outputTrain = torch.from_numpy(output_val_train).type(torch.LongTensor) 

inputTest = torch.from_numpy(input_val_test)
outputTest = torch.from_numpy(output_val_test).type(torch.LongTensor) 

# batch_size, epoch and iteration
batch_size = 100
iters = 10000
epochs = iters / (len(input_val_train) / batch_size)
epochs = int(epochs)

# Pytorch train and test sets
train_set = torch.utils.data.TensorDataset(inputTrain,outputTrain)
test_set = torch.utils.data.TensorDataset(inputTest,outputTest)

# data loader
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
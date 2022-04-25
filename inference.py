import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
from collections import OrderedDict
import copy
import json
import os
from os.path import exists
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir ='data/PlantVillage'
test_dir = data_dir + '/test'
train_dir =data_dir+'/train'
nThreads = 4
batch_size = 32
train = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
test = datasets.ImageFolder(test_dir, transform=transforms.ToTensor())
test_images = sorted(os.listdir(test_dir+'/test'))
print(test_images)

def load_trained_model():
    model = models.resnet101()
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 512)),('relu', nn.ReLU()),('fc2', nn.Linear(512,39)),('output', nn.LogSoftmax(dim=1))]))
    model.fc =classifier
    model.load_state_dict(torch.load('./plantsvillage_classifier_checkpoint.pth',map_location=torch.device(device)))
    model.eval()
    return model
     
      
    

def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    model.cuda()
    img =img.unsqueeze(0)
    xb = img.to(device)
    # to_device(img.unsqueeze(0), device)
    # Get predictions from model
    model.cuda()
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label

    return train.classes[preds[0].item()]
model =load_trained_model()
for i, (img, label) in enumerate(test):
    print('Label:', test_images[i], ', Predicted:', predict_image(img, model))

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
import pandas as pd


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir ='data/PlantVillage'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
nThreads = 4
batch_size =32
seed =7
weight_decay = 1e-4
df = pd.DataFrame(columns = ['Train loss'])
df = pd.DataFrame(columns = ['Validation loss'])
df = pd.DataFrame(columns = ['Train acc'])
df = pd.DataFrame(columns = ['Validation acc'])
df.to_csv('train_loss.csv',index=False)
df.to_csv('val_loss.csv',index=False)
df.to_csv('train_acc.csv',index=False)
df.to_csv('val_acc.csv',index=False)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
            ]),
    'val': transforms.Compose([
        transforms.Resize(224),       
        transforms.ToTensor(),       
    ]),
}

# Load the datasets with ImageFolder

data_dir = 'data/PlantVillage'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#np.random.seed(seed)
#torch.backends.cudnn.deterministic = True
#Resnet is feature extractor 
model = models.resnet101(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
#Replace the classification head with our custom classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(512, 39)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replacing the pretrained model classifier with our classifier
model.fc = classifier

#Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()             
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                       
                       
                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
            if phase=="train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            df = pd.DataFrame([epoch_loss])
            df.to_csv(phase+"_loss.csv", index=False, mode='a', header=False)
            df = pd.DataFrame([epoch_acc.cpu().item()])
            df.to_csv(phase+"_acc.csv", index=False, mode='a', header=False)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train a model with a pre-trained network
num_epochs = 20

model.to(device)

# NLLLoss because our output is LogSoftmax
criterion = nn.NLLLoss()

# Adam optimizer with a learning rate

optimizer = optim.Adam(model.fc.parameters(), lr=0.001,weight_decay=weight_decay)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)


# model inference
# Do validation on the test set
def test(model, dataloaders):
  model.eval()
  accuracy = 0
  
  model.to(device)
    
  for images, labels in dataloaders['val']:
    images = Variable(images)
    labels = Variable(labels)
    images, labels = images.to(device), labels.to(device)
      
    output = model.forward(images)
    ps = torch.exp(output)
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()
      
    print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders['val'])))


#run test model
test(model, dataloaders)

model.class_to_idx = dataloaders['train'].dataset.class_to_idx
model.epochs = num_epochs

torch.save(model.state_dict(),'plantsvillage_classifier_checkpoint.pth')


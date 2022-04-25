## **Plant Leaf Disease Detection using Convolutional Neural Networks**

The API allows the user to attach the URL of a plant leaf and the Deep Learning Model predicts if is it diseased.



#### Plants currently identified:
- Maize /Corn
- Apple
- Tomato
- Blue Berry
- Orange
- Grape
- Peach 
- Potatoe
- Cherry

### Dataset
This model was trained using the PlantVillage Dataset which is an open source plant diseases dataset.

### Model
 -  ResNet101 is used as the CNN feature extractor
 -  The classification head of ResNet101 was modified to accomodate the number class of prediction

### Training and hyper-parameter tuning
- Model was trained for approx. 6 hours on NVIDIA GeForce GTX-TITAN X GPU with 12GB of memory
- With learning rate of 0.001 for 20 epochs with batch size of 32
- Optimisation algorithm adopted was Adam with step size of 10

## Model Performance Metrics
Accuracy : 97%

### Data Preprocessing
- All images in the training dataset were resized to 224*224 to meet the input requirement of ResNet101
- Horizontal Flip was applied to as a data augmentation technique



### API
 
### Rest API's Created
#### Used pickle to save the ML model and Flask to provide the Frontend as well the API's.


| Method | API ROUTE | Actions |
|--|--| --|

| POST |https://frozen-chamber-77610.herokuapp.com/predict | Image to be tested is uploaded via a POST request and the predictions are returned. |

### Model Training
- Create a new directory with name data in the root directory 
- mkdir data/PlantVillage && cd data/PlantVillage
- Populate the PlantVillage directory with the train and validation dataset from the Plant Village website
- cd ..
- cp -r data/PlantVillage/test  ./
- cd ..&& cd ..
- run pip3 install -r requirements.txt (To install the dependencies)
- python3 trainer.py (Voila!)

### Testing
-run python3 inference.py

### Work in progress
- Improve the performance of the model by adding more data samples from other datasets
- Build a Mobile Application to make the model easily accessible
- Create a knowledge base to provide recommended treatment options for the identified diseases.




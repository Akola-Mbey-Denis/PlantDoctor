
import io
import json
from urllib.request import urlopen
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch
from flask import Flask, jsonify, request
from torch import nn
from collections import OrderedDict
app = Flask(__name__)
classes =['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
device=torch.device('cpu')
def load_trained_model():
    model = models.resnet101()
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 512)),('relu', nn.ReLU()),('fc2', nn.Linear(512,39)),('output', nn.LogSoftmax(dim=1))]))
    model.fc =classifier
    model.load_state_dict(torch.load('./plantsvillage_classifier_checkpoint.pth',map_location=torch.device(device)))
    model.eval()
    return model
      
model =load_trained_model()




def transform_image(image_bytes):
    my_transforms= transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),    
        transforms.ToTensor()])
         
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return classes[predicted_idx]
  


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        file_link = request.json['link']
        try:
          
            file = urlopen(file_link)
            img_bytes = file.read()
           
            class_name = get_prediction(image_bytes=img_bytes)
            return jsonify({'class_name': class_name})
        except:
            return "Try with a different image!"


if __name__ == '__main__':
    app.run()
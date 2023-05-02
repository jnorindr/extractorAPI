import io
import json

import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from flask import Flask, jsonify, request


# Application
app = Flask(__name__)
# Make sure to set `weights` as `'IMAGENET1K_V1'` to use the pretrained weights:
model = models.densenet121(weights='IMAGENET1K_V1')
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()
# Dictionnaire des index des classes alignés avec les noms ImageNet des classes
imagenet_class_index = json.load(open('imagenet_class_index.json'))


# Fonction transformant l'image en vecteurs (tensors)
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# Fonction retournant la classe ImageNet de l'objet identifié à partir de l'index obtenu à partir tensors
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()

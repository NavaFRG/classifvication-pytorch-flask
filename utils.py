import json
from PIL import Image
import torch
import onnx
import os
import onnxruntime
import io
import torchvision.transforms as transforms
import torchvision as tv
import torch

def get_model() :
    model_path = os.path.join('models', 'MedNet.onnx')
    model = onnx.load(model_path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')
    return model

def transform_image(image_bytes) :
    img = Image.open(io.BytesIO(image_bytes))
    img_y = scaleImage(img)
    img_y.unsqueeze_(0)
    return img_y

def format_class_name(class_name):
    class_name = class_name.title()
    return class_name

def scaleImage(x):
    toTensor = tv.transforms.ToTensor()
    y = toTensor(x)
    if(y.min() < y.max()):
        y = (y - y.min())/(y.max() - y.min())
    z = y - y.mean()
    return z

model_path = os.path.join('models', 'MedNet.onnx')
ort_session = onnxruntime.InferenceSession(model_path)
imagenet_class_index = json.load(open('imagenet_class_index.json'))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_prediction(image_bytes):
    try:
        img_y = transform_image(image_bytes)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[0]
    except Exception:
        return 404, 'error'
    return {imagenet_class_index.get(str(outputs.argmax())),outputs.argmax()}

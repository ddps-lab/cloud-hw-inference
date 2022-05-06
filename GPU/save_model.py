import json
import boto3
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications import (
    xception,
    vgg16,
    vgg19,
    resnet,
    resnet50,
    resnet_v2,
    inception_v3,
    inception_resnet_v2,
    mobilenet,
    densenet,
    nasnet,
    mobilenet_v2,
    efficientnet
)
from tensorflow.keras.models import load_model, save_model
import tensorflow.compat.v1.keras as keras
import sys

models_detail = {
    'xception': xception.Xception(weights='imagenet'),
    'vgg16': vgg16.VGG16(weights='imagenet'),
    'vgg19': vgg19.VGG19(weights='imagenet'),
    'resnet50': resnet50.ResNet50(weights='imagenet'),
    'resnet101': resnet.ResNet101(weights='imagenet'),
    'resnet152': resnet.ResNet152(weights='imagenet'),
    'resnet50_v2': resnet_v2.ResNet50V2(weights='imagenet'),
    'resnet101_v2': resnet_v2.ResNet101V2(weights='imagenet'),
    'resnet152_v2': resnet_v2.ResNet152V2(weights='imagenet'),
    'inception_v3': inception_v3.InceptionV3(weights='imagenet'),
    'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'mobilenet': mobilenet.MobileNet(weights='imagenet'),
    'densenet121': densenet.DenseNet121(weights='imagenet'),
    'densenet169': densenet.DenseNet169(weights='imagenet'),
    'densenet201': densenet.DenseNet201(weights='imagenet'),
    'nasnetmobile': nasnet.NASNetLarge(weights='imagenet'),
    'nasnetlarge': nasnet.NASNetMobile(weights='imagenet'),
    'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet'),
    'efficientnetb1': efficientnet.EfficientNetB1(weights='imagenet'),
    'efficientnetb2': efficientnet.EfficientNetB2(weights='imagenet'),
    'efficientnetb3': efficientnet.EfficientNetB3(weights='imagenet'),
    'efficientnetb4': efficientnet.EfficientNetB4(weights='imagenet'),
    'efficientnetb5': efficientnet.EfficientNetB5(weights='imagenet'),
    'efficientnetb6': efficientnet.EfficientNetB6(weights='imagenet'),
    'efficientnetb7': efficientnet.EfficientNetB7(weights='imagenet')
}

model_type = str(sys.argv[1])
saved_model_dir = f'model/{model_type}'

model = models_detail[model_type]
model.save(saved_model_dir)

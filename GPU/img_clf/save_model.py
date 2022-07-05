# saved pretrained tensorflow model

import tensorflow as tf
import argparse
import os 
from tensorflow.keras.applications import (
    xception,
    vgg16,
    resnet50,
    inception_v3,
    mobilenet_v2
)

models_detail = {
    'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet'),
    'resnet50': resnet50.ResNet50(weights='imagenet'),
    'inception_v3': inception_v3.InceptionV3(weights='imagenet'),
    'xception': xception.Xception(weights='imagenet'),
    'vgg16': vgg16.VGG16(weights='imagenet'),
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str , required=True)

    model_name = parser.parse_args().model
    
    saved_model_dir = f'{model_name}'
    model = models_detail[model_name]
    model.save(saved_model_dir) 
    print(f"Pretrain {model_name} saved in {saved_model_dir}")

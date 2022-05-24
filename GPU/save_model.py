# saved pretrained tensorflow model: tensorflow==2.3.0

import tensorflow as tf
import argparse
import os 
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
    mobilenet_v2
)
from tensorflow.keras.models import load_model, save_model


models_detail = {
    'mobilenet': mobilenet.MobileNet(weights='imagenet'),
    'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet'),
    'resnet50': resnet50.ResNet50(weights='imagenet'),
    'resnet101': resnet.ResNet101(weights='imagenet'),
    'resnet152': resnet.ResNet152(weights='imagenet'),
    'resnet50_v2': resnet_v2.ResNet50V2(weights='imagenet'),
    'resnet101_v2': resnet_v2.ResNet101V2(weights='imagenet'),
    'resnet152_v2': resnet_v2.ResNet152V2(weights='imagenet'),
    'inception_v3': inception_v3.InceptionV3(weights='imagenet'),
    'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'densenet121': densenet.DenseNet121(weights='imagenet'),
    'densenet169': densenet.DenseNet169(weights='imagenet'),
    'densenet201': densenet.DenseNet201(weights='imagenet'),
    'nasnetmobile': nasnet.NASNetLarge(weights='imagenet'),
    'nasnetlarge': nasnet.NASNetMobile(weights='imagenet'),
    'xception': xception.Xception(weights='imagenet'),
    'vgg16': vgg16.VGG16(weights='imagenet'),
    'vgg19': vgg19.VGG19(weights='imagenet'),
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str , required=True)

    model_name = parser.parse_args().model
    
    saved_model_dir = f'{model_name}'
    model = models_detail[model_name]
    model.save(saved_model_dir) 
    print(f"Pretrain {model_name} saved in {saved_model_dir}")

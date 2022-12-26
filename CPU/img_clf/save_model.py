# saved pretrained tensorflow model
import argparse
from tensorflow.keras.applications import ( 
#     xception,
#     vgg16,
#     vgg19,
#     resnet,
#     resnet50,
#     resnet_v2,
#     inception_v3,
#     inception_resnet_v2,
#     mobilenet,
#     densenet,
#     nasnet,
#     mobilenet_v2,
    efficientnet,
    efficientnet_v2,
#     mobilenet_v3,
    MobileNetV3Small,
    MobileNetV3Large,
)

from tensorflow.keras.layers import Input
input_tensor = Input(shape=(224, 224, 3))

models_detail = {
#     'xception':xception.Xception(weights='imagenet'),
#     'vgg16':vgg16.VGG16(weights='imagenet'),
#     'resnet50':resnet50.ResNet50(weights='imagenet'),
#     'resnet101':resnet.ResNet101(weights='imagenet'),
#     'resnet152':resnet.ResNet152(weights='imagenet'),
#     'resnet50_v2':resnet_v2.ResNet50V2(weights='imagenet'),
#     'resnet101_v2':resnet_v2.ResNet101V2(weights='imagenet'),
#     'resnet152_v2':resnet_v2.ResNet152V2(weights='imagenet'),
#     'inception_v3':inception_v3.InceptionV3(weights='imagenet'),
#     'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
#     'mobilenet':mobilenet.MobileNet(weights='imagenet'),
#     'densenet121':densenet.DenseNet121(weights='imagenet'),
#     'densenet169':densenet.DenseNet169(weights='imagenet'),
#     'densenet201':densenet.DenseNet201(weights='imagenet'),
#     'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
#     'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
#     'efficientnetb0':efficientnet.EfficientNetB0(weights='imagenet'),
#     'efficientnetb1':efficientnet.EfficientNetB1(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnetb2':efficientnet.EfficientNetB2(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnetb3':efficientnet.EfficientNetB3(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnetb4':efficientnet.EfficientNetB4(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnetb5':efficientnet.EfficientNetB5(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnetb6':efficientnet.EfficientNetB6(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnetb7':efficientnet.EfficientNetB7(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnet_v2b0':efficientnet_v2.EfficientNetV2B0(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnet_v2b1':efficientnet_v2.EfficientNetV2B1(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnet_v2b2':efficientnet_v2.EfficientNetV2B2(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnet_v2b3':efficientnet_v2.EfficientNetV2B3(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'efficientnet_v2l':efficientnet_v2.EfficientNetV2L(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2m':efficientnet_v2.EfficientNetV2M(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2s':efficientnet_v2.EfficientNetV2S(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'mobilenet_v3small':MobileNetV3Small(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'mobilenet_v3large':MobileNetV3Large(input_tensor=input_tensor, weights='imagenet', include_top=True),
#     'vgg19':vgg19.VGG19(weights='imagenet'),
#     'nasnetlarge':nasnet.NASNetLarge(input_tensor=input_tensor, weights='imagenet', include_top=True),
}

model_types = [key for key, value in models_detail.items()]

print(model_types)

if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model',default='resnet50' , type=str , required=True)
    
#     model_name = parser.parse_args().model
    for model_name in model_types:
        saved_model_dir = f'{model_name}'
        model = models_detail[model_name]
        model.save(saved_model_dir) 
        print(f"Pretrain {model_name} saved in {saved_model_dir}")

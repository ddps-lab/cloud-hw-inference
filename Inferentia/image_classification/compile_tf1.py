import os
import time
import shutil
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras
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
    efficientnet,
    efficientnet_v2,
    mobilenet_v3,
    MobileNetV3Small,
    MobileNetV3Large,
)
# from keras import backend as K
from tensorflow.keras.preprocessing import image
# from concurrent import futures
from itertools import compress
from tensorflow.keras.models import load_model

models = {
#     'xception':xception,
#     'vgg16':vgg16,
    'vgg19':vgg19,
#     'resnet50':resnet50,
    'resnet101':resnet,
    'resnet152':resnet,
    'resnet50_v2':resnet_v2,
    'resnet101_v2':resnet_v2,
    'resnet152_v2':resnet_v2,
#     'inception_v3':inception_v3,
    'inception_resnet_v2':inception_resnet_v2,
    'mobilenet':mobilenet,
    'densenet121':densenet,
    'densenet169':densenet,
    'densenet201':densenet,
    'nasnetmobile':nasnet,
    'nasnetlarge':nasnet,
#     'mobilenet_v2':mobilenet_v2
    'efficientnetb0':efficientnet,
    'efficientnetb1':efficientnet,
    'efficientnetb2':efficientnet,
    'efficientnetb3':efficientnet,
    'efficientnetb4':efficientnet,
    'efficientnetb5':efficientnet,
    'efficientnetb6':efficientnet,
    'efficientnetb7':efficientnet,
    'efficientnet_v2b0':efficientnet_v2,
    'efficientnet_v2b1':efficientnet_v2,
    'efficientnet_v2b2':efficientnet_v2,
    'efficientnet_v2b3':efficientnet_v2,
    'efficientnet_v2l':efficientnet_v2,
    'efficientnet_v2m':efficientnet_v2,
    'efficientnet_v2s':efficientnet_v2,
    'mobilenet_v3small':mobilenet_v3,
    'mobilenet_v3large':mobilenet_v3,
}

models_detail = {
#     'xception':xception.Xception(weights='imagenet'),
#     'vgg16':vgg16.VGG16(weights='imagenet'),
    'vgg19':vgg19.VGG19(weights='imagenet'),
#     'resnet50':resnet50.ResNet50(weights='imagenet'),
    'resnet101':resnet.ResNet101(weights='imagenet'),
    'resnet152':resnet.ResNet152(weights='imagenet'),
    'resnet50_v2':resnet_v2.ResNet50V2(weights='imagenet'),
    'resnet101_v2':resnet_v2.ResNet101V2(weights='imagenet'),
    'resnet152_v2':resnet_v2.ResNet152V2(weights='imagenet'),
#     'inception_v3':inception_v3.InceptionV3(weights='imagenet'),
    'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'mobilenet':mobilenet.MobileNet(weights='imagenet'),
    'densenet121':densenet.DenseNet121(weights='imagenet'),
    'densenet169':densenet.DenseNet169(weights='imagenet'),
    'densenet201':densenet.DenseNet201(weights='imagenet'),
    'nasnetlarge':nasnet.NASNetLarge(weights='imagenet'),
    'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
#     'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
    'efficientnetb0':efficientnet.EfficientNetB0(weights='imagenet'),
    'efficientnetb1':efficientnet.EfficientNetB1(weights='imagenet'),
    'efficientnetb2':efficientnet.EfficientNetB2(weights='imagenet'),
    'efficientnetb3':efficientnet.EfficientNetB3(weights='imagenet'),
    'efficientnetb4':efficientnet.EfficientNetB4(weights='imagenet'),
    'efficientnetb5':efficientnet.EfficientNetB5(weights='imagenet'),
    'efficientnetb6':efficientnet.EfficientNetB6(weights='imagenet'),
    'efficientnetb7':efficientnet.EfficientNetB7(weights='imagenet'),
    'efficientnet_v2b0':efficientnet_v2.EfficientNetV2B0(weights='imagenet'),
    'efficientnet_v2b1':efficientnet_v2.EfficientNetV2B1(weights='imagenet'),
    'efficientnet_v2b2':efficientnet_v2.EfficientNetV2B2(weights='imagenet'),
    'efficientnet_v2b3':efficientnet_v2.EfficientNetV2B3(weights='imagenet'),
    'efficientnet_v2l':efficientnet_v2.EfficientNetV2L(weights='imagenet'),
    'efficientnet_v2m':efficientnet_v2.EfficientNetV2M(weights='imagenet'),
    'efficientnet_v2s':efficientnet_v2.EfficientNetV2S(weights='imagenet'),
    'mobilenet_v3small':MobileNetV3Small(weights='imagenet'),
    'mobilenet_v3large':MobileNetV3Large(weights='imagenet'),
}

model_neuron_ratios = []

model_types = [key for key, value in models_detail.items()]

print(model_types)

# for model_type in model_types:
#     # https://github.com/tensorflow/tensorflow/issues/29931
#     temp = tf.zeros([8, 224, 224, 3])
#     _ = models[model_type].preprocess_input(temp)

#     # Export SavedModel

#     saved_model_dir = f'{model_type}_saved_model'
#     shutil.rmtree(saved_model_dir, ignore_errors=True)

#     model = models_detail[model_type]

#     model.save(saved_model_dir)

#     from tensorflow.keras.models import load_model
#     model = load_model(saved_model_dir, compile=True)

#     model.summary()

    
def compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=1, num_cores=1, use_static_weights=False):
    print(f'-----------batch size: {batch_size}----------')
    print('Compiling...')
    
    compiled_model_dir = f'{model_type}_batch_{batch_size}'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
    shutil.rmtree(inf1_compiled_model_dir, ignore_errors=True)

    example_input = np.zeros([batch_size,224,224,3], dtype='float32')
    if "xception" in saved_model_dir or "inception_v3" in saved_model_dir:
        example_input = np.zeros([batch_size,299,299,3], dtype='float32')
        
    model = load_model(saved_model_dir, compile=True)
    
    start_time = time.time()
    compiled_res = tfn.saved_model.compile(saved_model_dir, compiled_model_dir)
    print(f'Compile time: {time.time() - start_time}')
    
    compile_success = False
    perc_on_inf = compiled_res['OnNeuronRatio'] * 100
    if perc_on_inf > 50:
        compile_success = True
            
    print(inf1_compiled_model_dir)
    print(compiled_res)
    model_neuron_ratios.append(prec_on_inf)
    print('----------- Done! ----------- \n')
    
    return compile_success


for model_type in model_types:
    inf1_model_dir = f'{model_type}_inf1_saved_models'
    saved_model_dir = f'{model_type}_saved_model'


    # testing batch size
    batch_list = [1]
    for batch in batch_list:
        print('batch size:', batch,'compile start')
        compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=batch)
        
print(model_neuron_ratios)

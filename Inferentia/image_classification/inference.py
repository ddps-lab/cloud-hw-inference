import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
    mobilenet_v2
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from concurrent import futures
from itertools import compress

models = {
    'xception':xception,
    'vgg16':vgg16,
#     'vgg19':vgg19,
    'resnet50':resnet50,
#     'resnet101':resnet,
#     'resnet152':resnet,
#     'resnet50_v2':resnet_v2,
#     'resnet101_v2':resnet_v2,
#     'resnet152_v2':resnet_v2,
#     'resnext50':resnext,
#     'resnext101':resnext,
    'inception_v3':inception_v3,
#     'inception_resnet_v2':inception_resnet_v2,
#     'mobilenet':mobilenet,
#     'densenet121':densenet,
#     'densenet169':densenet,
#     'densenet201':densenet,
#     'nasnetlarge':nasnet,
#     'nasnetmobile':nasnet,
    'mobilenet_v2':mobilenet_v2
}

data_dir = os.environ['dataset']

def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1)}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    return imgdata, label

def val_preprocessing(record):
    imgdata, label = deserialize_image_record(record)
    label -= 1
    image = tf.io.decode_jpeg(imgdata, channels=3, 
                              fancy_upscaling=False, 
                              dct_method='INTEGER_FAST')

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)

    scale = tf.cond(tf.greater(height, width),
                  lambda: side / width,
                  lambda: side / height)
    
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    
    image = models[model_type].preprocess_input(image)
    
    return image, label

def get_dataset(batch_size, use_cache=False):
#     files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(data_dir)
    print('files',dataset)
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset
  
import os
    
def inf1_predict_benchmark_single_threaded(neuron_saved_model_name, batch_size, user_batch_size, num_cores, use_cache=False, warm_up=10):
    print(f'Running model {neuron_saved_model_name}, user_batch_size: {user_batch_size}\n')

    model_inf1 = load_model(neuron_saved_model_name)
    
    inference_function = model_inf1.signatures['serving_default']
    walltime_start = time.time()
    first_iter_time = 0
    iter_times = []
    pred_labels = []
    actual_labels = []
    total_datas = 50000
    display_every = 1000
    display_threshold = display_every
    
    ds = get_dataset(user_batch_size, use_cache)
    load_start = time.time()
    load_time = time.time() - load_start
    counter = 0
    print(model_inf1)
    for batch, batch_labels in ds:
        print('batch',batch)
        print(batch_labels)
        start_time = time.time()
        yhat_np = inference_function(batch)
        if counter ==0:
            first_iter_time = time.time() - start_time
        else:
            iter_times.append(time.time() - start_time)
        actual_labels.extend(label for label_list in batch_labels for label in label_list)
        pred_labels.extend(list(np.argmax(yhat_np, axis=1)))

        if counter*batch_size >= display_threshold:
            print(f'Images {counter*batch_size}/{total_datas}. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every

        counter+=1
    iter_times = np.array(iter_times)
    print('actual',actual_labels)
    print(pred_labels)
    acc_inf1 = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    results = pd.DataFrame(columns = [f'inf1_tf2_{model_type}_{batch_size}'])
    results.loc['batch_size']              = [batch_size]
    results.loc['accuracy']                = [acc_inf1]
    results.loc['first_prediction_time']   = [first_iter_time]
    results.loc['average_prediction_time'] = [np.mean(iter_times)]
    results.loc['load_time']               = [load_time]
    results.loc['wall_time']               = [time.time() - walltime_start]

    return results, iter_times
  
model_types = ['resnet50', 'vgg16', 'xception', 'inception_v3', 'mobilenet_v2']

for model_type in model_types:
    # https://github.com/tensorflow/tensorflow/issues/29931
    temp = tf.zeros([8, 224, 224, 3])
    _ = models[model_type].preprocess_input(temp)

    # testing batch size
    batch_list = [1]
    num_of_cores = [1]
    user_batchs = [1]
    inf1_model_dir = f'{model_type}_inf1_saved_models'

    for user_batch in user_batchs:
        iter_ds = pd.DataFrame()
        results = pd.DataFrame()
        for batch_size in batch_list:
            for num_cores in num_of_cores:
                opt ={'batch_size': batch_size, 'num_cores': num_of_cores}
                compiled_model_dir = f'{model_type}_batch_{batch_size}_inf1_cores_{num_cores}'
                inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)

                print(f'inf1_compiled_model_dir: {inf1_compiled_model_dir}')
                col_name = lambda opt: f'inf1_{batch_size}_multicores_{num_cores}'

                res, iter_times = inf1_predict_benchmark_single_threaded(inf1_compiled_model_dir,
                                                                                 batch_size = batch_size,
                                                                                 user_batch_size = batch_size*user_batch,
                                                                                 num_cores = num_cores,
                                                                                 use_cache=False, 
                                                                                 warm_up=10)

            iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
            results = pd.concat([results, res], axis=1)
        results.to_csv(f'{model_type}_batch_size_{batch_size}.csv')

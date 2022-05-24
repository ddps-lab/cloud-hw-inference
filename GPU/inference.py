import os, time
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
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
#     efficientnet
)
import sys
import argparse

models = {
    'xception':xception,
    'vgg16':vgg16,
    'vgg19':vgg19,
    'resnet50':resnet50,
    'resnet101':resnet,
    'resnet152':resnet,
    'resnet50_v2':resnet_v2,
    'resnet101_v2':resnet_v2,
    'resnet152_v2':resnet_v2,
    'inception_v3':inception_v3,
    'inception_resnet_v2':inception_resnet_v2,
    'mobilenet':mobilenet,
    'densenet121':densenet,
    'densenet169':densenet,
    'densenet201':densenet,
    'nasnetlarge':nasnet,
    'nasnetmobile':nasnet,
    'mobilenet_v2':mobilenet_v2,
#     'efficientnet_b7':efficientnet
}


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
    if "inception" in model or "xception" in model:
        image = tf.image.resize_with_crop_or_pad(image, 299, 299)
    else:
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    label = tf.cast(label, tf.int32)
    image = models[model].preprocess_input(image)
    image = tf.cast(image, tf.float32)
    return image, label

def get_dataset(batch_size, use_cache=False):
    data_dir = '/home/ubuntu/datasets/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset

def inference(saved_model_name, batch_size):
    walltime_start = time.time()
    first_iter_time = 0
    iter_times = []
    pred_labels = []
    actual_labels = []
    total_datas = 50000
    display_every = 1000
    display_threshold = display_every
    warm_up = 10
    
    ds = get_dataset(batch_size)

    load_start = time.time()
    model = load_model(saved_model_name)
    load_time = time.time() - load_start
    counter = 0

    with tf.device("/device:GPU:0"):
        for batch, batch_labels in ds:
            if counter == 0:
                for i in range(warm_up):
                    _ = model.predict(batch)

            start_time = time.time()
            yhat_np = model.predict(batch)

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
            
            if counter == 100:
                break

        iter_times = np.array(iter_times)
        acc_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)

        results = pd.DataFrame(columns = [f'GPU_{saved_model_name}_{batch_size}'])
        results.loc['batch_size']              = [batch_size]
        results.loc['accuracy']                = [acc_gpu]
        results.loc['prediction_time']         = [np.sum(iter_times)*1000]
        results.loc['images_per_sec_mean']     = [np.mean(batch_size / iter_times)]
        results.loc['first_iteration_time']   = [first_iter_time * 1000]
        results.loc['average_iteration_time'] = [np.mean(iter_times[1:]) * 1000]
        results.loc['load_time']               = [load_time*1000]
        results.loc['wall_time']               = [(time.time() - walltime_start)*1000]

        return results, iter_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str )
    parser.add_argument('--batch_list',default=[1,8,16,32,64,128,256,512], type=list)

    model = parser.parse_args().model
    batch_list = parser.parse_args().batch_list

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    
    results = pd.DataFrame()

    for batch_size in batch_list:
        opt = {'batch_size': batch_size}
        iter_ds = pd.DataFrame()
        
        print(f'{model}-{batch_size} start')
        res, iter_times = inference(model, int(batch_size))
        col_name = lambda opt: f'{model}_{batch_size}'
        
        iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
        results = pd.concat([results, res], axis=1)
        print(results)
        results.to_csv(f'{model}.csv')

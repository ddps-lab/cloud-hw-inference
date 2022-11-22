import tensorflow as tf
from ei_for_tf.python.predictor.ei_predictor import EIPredictor
import numpy as np
import pandas as pd
import shutil
import requests
import time
import os
import boto3
import argparse

print(tf.__version__) 

# # https://github.com/tensorflow/tensorflow/issues/29931
# temp = tf.zeros([1, 224, 224, 3])
# _ = tf.keras.applications.resnet50.preprocess_input(temp)

ei_client = boto3.client('elastic-inference',region_name='us-west-2')

# Dataset processing

def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                  'image/class/text': tf.io.FixedLenFeature([], tf.string, '')}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    label_text = tf.cast(obj['image/class/text'], tf.string)   
    return imgdata, label, label_text

def val_preprocessing(record):
    imgdata, label, label_text = deserialize_image_record(record)
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
    
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batch_size, use_cache=False):
    data_dir = '/home/ec2-user/datasets/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    if use_cache:
        shutil.rmtree('tfdatacache', ignore_errors=True)
        os.mkdir('tfdatacache')
        dataset = dataset.cache(f'./tfdatacache/imagenet_val')
    
    return dataset


def ei_inference(saved_model_dir, batch_size, accelerator_id):
    
    ei_size = ei_client.describe_accelerators()['acceleratorSet'][accelerator_id]['acceleratorType']

    print('\n=======================================================')
    print(f'Benchmark results for EI: {ei_size}, batch size: {batch_size}')
    print('=======================================================\n')
    

    display_every = 1000
    warm_up = 10
    display_threshold = display_every


    pred_labels = []
    actual_labels = []
    iter_times = []

    load_start=time.time()
    eia_model = EIPredictor(saved_model_dir, 
                                accelerator_id=accelerator_id)
    load_time = time.time()-load_start
    print("Make EIA model time : ",load_time*1000,"ms")

    # Get the tf.data.TFRecordDataset object for the ImageNet2012 validation dataset
    dataset = get_dataset(batch_size)  

    walltime_start = time.time()
    ipname = list(eia_model.feed_tensors.keys())[0]
    resname = list(eia_model.fetch_tensors.keys())[0]
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        model_feed_dict={f'{ipname}': validation_ds.numpy()}

        if i == 0:
            for j in range(warm_up):
                _ = eia_model(model_feed_dict)
        tf.profiler.experimental.start('profile')
        start_time = time.time()
        
        pred_prob = eia_model(model_feed_dict)
        inf_time = time.time() - start_time
        tf.profiler.experimental.stop()

        if i ==0:
            first_iter_time = inf_time
        else:
            iter_times.append(inf_time)

        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(np.argmax(pred_prob['predictions'], axis=1)))

        if i*batch_size >= display_threshold:
            print(f'Images {i*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every

    iter_times = np.array(iter_times)
    acc_ei_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    
    results = pd.DataFrame(columns = [f'EIA_{saved_model_dir}_{batch_size}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['accelerator']             = [ei_size]
    results.loc['user_batch_size']         = [batch_size]
    results.loc['accuracy']                = [acc_ei_gpu]
    results.loc['total_inference_time']    = [np.sum(iter_times)*1000]
    results.loc['images_per_sec_mean']     = [np.mean(batch_size / iter_times)]
    results.loc['first_iteration_time']    = [first_iter_time * 1000]
    results.loc['average_iteration_time']  = [np.mean(iter_times[1:]) * 1000]
    results.loc['latency_median']          = [np.median(iter_times) * 1000]
    results.loc['latency_99th_percentile'] = [np.percentile(iter_times, q=99, interpolation="lower") * 1000]
    results.loc['load_time']               = [load_time*1000]
    results.loc['wall_time']               = [(time.time() - walltime_start)*1000]
    
    return results, iter_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str )
    parser.add_argument('--imgsize',default=224,type=int)
    parser.add_argument('--batch',default=1, type=int)
    parser.add_argument('--eia_acc_id',default=0,type=int) 
    
    model = parser.parse_args().model
    batch_size = parser.parse_args().batch
    eia_acc_id = parser.parse_args().eia_acc_id
    img_size = parser.parse_args().imgsize

    results = pd.DataFrame()
    ei_size = ei_client.describe_accelerators()['acceleratorSet'][eia_acc_id]['acceleratorType']

    col_name = lambda ei_acc_id: f'ei_{ei_client.describe_accelerators()["acceleratorSet"][ei_acc_id]["acceleratorType"]}_batch_size_{batch_size}'
    # ei_options = [{'ei_acc_id': 0}]
    opt = {'batch_size': batch_size}
    iter_ds = pd.DataFrame()

    print(f'{model}-{batch_size} start')
    res, iter_times = ei_inference(model, int(batch_size),eia_acc_id)
    col_name = lambda opt: f'{model}_{batch_size}'

    iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
    results = pd.concat([results, res], axis=1)
    print(results)

    results.to_csv(f'{model}_EIA_{ei_size}.csv')

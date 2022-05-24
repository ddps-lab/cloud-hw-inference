import os
import time
import numpy as np
import pandas as pd
import shutil
import requests
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants, signature_constants


from tensorflow.keras.applications import ( 
    vgg16,
    vgg19,
    resnet,
    resnet50,
    resnet_v2,
    inception_v3,
    inception_resnet_v2,
    mobilenet,
    mobilenet_v2,
    densenet,
    nasnet,
    xception,
)
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
    'mobilenet_v2':mobilenet_v2
}
models_detail = {
    'xception':xception.Xception(weights='imagenet',include_top=False),
    'vgg16':vgg16.VGG16(weights='imagenet'),
    'vgg19':vgg19.VGG19(weights='imagenet'),
    'resnet50':resnet50.ResNet50(weights='imagenet'),
    'resnet101':resnet.ResNet101(weights='imagenet'),
    'resnet152':resnet.ResNet152(weights='imagenet'),
    'resnet50_v2':resnet_v2.ResNet50V2(weights='imagenet'),
    'resnet101_v2':resnet_v2.ResNet101V2(weights='imagenet'),
    'resnet152_v2':resnet_v2.ResNet152V2(weights='imagenet'),
    'inception_v3':inception_v3.InceptionV3(weights='imagenet',include_top=False),
    'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'mobilenet':mobilenet.MobileNet(weights='imagenet'),
    'densenet121':densenet.DenseNet121(weights='imagenet'),
    'densenet169':densenet.DenseNet169(weights='imagenet'),
    'densenet201':densenet.DenseNet201(weights='imagenet'),
    'nasnetlarge':nasnet.NASNetLarge(weights='imagenet'),
    'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
    'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet')
}

import argparse

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--batchsize',default=1,type=int)
parser.add_argument('--precision',default='FP32',type=str)
parser.add_argument('--load',default=False,type=bool)
parser.add_argument('--gpu',default=False,type=bool)
parser.add_argument('--engines',default=64,type=int)
args = parser.parse_args()
load_model = args.model
batch_size = args.batchsize
precision = args.precision
load=args.load
run_gpu=args.gpu
num_engine=args.engines


def load_save_model(load_model , saved_model_dir = 'resnet50_saved_model'):

    model = models_detail[load_model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    model.save(saved_model_dir, save_format='tf')

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
    
    image = models[load_model].preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batch_size, use_cache=False):
    data_dir = '../datasets/images-1000/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset

def calibrate_fn(n_calib, batch_size, dataset):
    for i, (calib_image, _, _) in enumerate(dataset):
        if i > n_calib // batch_size:
            break
        yield (calib_image,)

def build_fn(batch_size, dataset):
    for i, (build_image, _, _) in enumerate(dataset):
        if i > 1:
            break
        yield (build_image,)

def build_FP_tensorrt_engine(load_model,precision, batch_size, dataset):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    if precision == 'FP32':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP32,
                                                        max_workspace_size_bytes=8000000000,maximum_cached_engines=num_engine)
    elif precision == 'FP16':                                                 
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP16,
                                                        max_workspace_size_bytes=8000000000,maximum_cached_engines=num_engine)
    
    elif precision=='INT8':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.INT8, 
                                                        max_workspace_size_bytes=8000000000, 
                                                        use_calibration=True ,maximum_cached_engines=num_engine )
    #conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision.upper(),
    #                                                               max_workspace_size_bytes=(1<<32),
    #                                                               maximum_cached_engines=2)
    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=load_model,
                                        conversion_params=conversion_params)
    
    if precision=='INT8':
        #converter.convert(calibration_input_fn=calibration_input_fn)
        n_calib=50
        converter.convert(calibration_input_fn=partial(calibrate_fn, n_calib, batch_size, 
                                                       dataset.shuffle(buffer_size=n_calib, reshuffle_each_iteration=True)))
    else:
        converter.convert()
        
    trt_compiled_model_dir = f'{load_model}_{precision}'
    #converter.save(output_saved_model_dir=trt_compiled_model_dir)

    # shutil.rmtree(trt_compiled_model_dir, ignore_errors=True)
    print("TensorEngine Build")
    converter.build(input_fn=partial(build_fn, batch_size, dataset))
    print("tensorRT_SAVED")
    converter.save(output_saved_model_dir=trt_compiled_model_dir)
    print(f'\nOptimized for {precision} and batch size {batch_size}, directory:{trt_compiled_model_dir}\n')

    return trt_compiled_model_dir

def trt_predict_benchmark(trt_compiled_model_dir,precision, batch_size, use_cache=False, display_every=100, warm_up=50):

    print('\n=======================================================')
    print(f'Benchmark results for precision: {precision}, batch size: {batch_size}')
    print('=======================================================\n')
    
    dataset = get_dataset(batch_size)
    
    # If caching is enabled, cache dataset for better i/o performance
    if counter == 0:
        for i in range(warm_up):
            _ = model.predict(batch)

    # saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    # trt_compiled_model_dir = build_FP_tensorrt_engine(precision, batch_size, dataset)
    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures['serving_default']

    pred_labels = []
    actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every
    initial_time = time.time()

    walltime_start = time.time()
    N=0
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        N+=1
        if i==0:
            for w in range(warm_up):
                _ = model_trt(validation_ds);
                
        start_time = time.time()
        trt_results = model_trt(validation_ds);
        iter_times.append(time.time() - start_time)
        
        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(tf.argmax(trt_results['predictions'], axis=1).numpy()))
        if (i)*batch_size >= display_threshold:
            print(f'Images {(i)*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every
    
    print('Throughput: {:.0f} images/s'.format(N * batch_size / sum(iter_times)))

    acc_trt = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    iter_times = np.array(iter_times)
   
    results = pd.DataFrame(columns = [f'trt_{precision}_{batch_size}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['user_batch_size']         = [batch_size]
    results.loc['accuracy']                = [acc_trt]
    results.loc['prediction_time']         = [np.sum(iter_times)]
    results.loc['wall_time']               = [time.time() - walltime_start]   
    results.loc['images_per_sec_mean']     = [np.mean(batch_size / iter_times)]
    results.loc['images_per_sec_std']      = [np.std(batch_size / iter_times, ddof=1)]
    results.loc['latency_mean']            = [np.mean(iter_times) * 1000]
    results.loc['latency_99th_percentile'] = [np.percentile(iter_times, q=99, interpolation="lower") * 1000]
    results.loc['latency_median']          = [np.median(iter_times) * 1000]
    results.loc['latency_min']             = [np.min(iter_times) * 1000]
    results.loc['first_batch']             = [iter_times[0]]
    results.loc['next_batches_mean']       = [np.mean(iter_times[1:])]
    print(results)
   
    return results, iter_times

saved_model_dir = load_model

if load :
    load_save_model(load_model,saved_model_dir)

dataset = get_dataset(batch_size)


print("------TENSORRT-----")
trt_compiled_model_dir = build_FP_tensorrt_engine(load_model,precision, batch_size, dataset)
print("------TENSORRT_INFERENCE-------")
trt_predict_benchmark(trt_compiled_model_dir,precision, batch_size, use_cache=False, display_every=100, warm_up=10)


import time
import pandas as pd
import numpy as np
import requests
import argparse

import tensorflow as tf

from tensorflow.python.saved_model import tag_constants, signature_constants

print(f"TensorFlow version: {tf.__version__}")


results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50',type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--precision',default='fp32', type=str)
parser.add_argument('--size',default=224,type=int)
args = parser.parse_args()
model = args.model 
batch_size = args.batchsize
model_batchsize=args.load_batchsize
precision = args.precision
size=args.size




def trt_predict_benchmark(precision, batch_size,imgsz, display_every=100, warm_up=10,repeat=10):

    print('\n=======================================================')
    print(f'Benchmark results for precision: {precision}, batch size: {batch_size}')
    print('=======================================================\n')
    
   
    # LOAD 만 해서 inference 진행
    model_name="yolov5s"
    trt_compiled_model_dir = f'{model_name}_saved_model_trt_saved_models/{model_name}_saved_model_fp32_1'
    # trt_compiled_model_dir = build_tensorrt_engine(precision, batch_size, dataset)

    walltime_start = time.time()

    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        
    pred_labels = []
    # actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every
    initial_time = time.time()

    image_shape = (imgsz, imgsz,3)
    # image_shape = (3, imgsz, imgsz)
    data_shape = (batch_size,) + image_shape
    
    for i in range(repeat):
        img = np.random.uniform(-1, 1 , size=data_shape).astype("float32")

        if i==0:
            for w in range(warm_up):
                _ = model_trt(x=img);
                
        start_time = time.time()
        trt_results = model_trt(x=img);
        iter_times.append(time.time() - start_time)
        
        # actual_labels.extend(label for label_list in labels.numpy() for label in label_list)
        
        # print(trt_results)
        pred_labels.extend(list(tf.argmax(trt_results['output_0'], axis=1).numpy()))
        if (i)*batch_size >= display_threshold:
            print(f'Images {(i)*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every
    
    print(f'Wall time: {time.time() - walltime_start}')

    iter_times = np.array(iter_times)
   
    results = pd.DataFrame(columns = [f'trt_{precision}_{batch_size}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['user_batch_size']         = [batch_size]
    # results.loc['accuracy']                = [acc_trt]
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
    print(results.T)
   
    return results, iter_times


if results is None:
    results = pd.DataFrame()

imgsz = 640
res, it = trt_predict_benchmark(precision, batch_size,imgsz)
results = res

print(results)
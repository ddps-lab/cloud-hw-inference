import time
import pandas as pd
import numpy as np
import requests
import argparse

import tensorflow as tf

from tensorflow.python.saved_model import tag_constants, signature_constants

print(f"TensorFlow version: {tf.__version__}")

def trt_predict_benchmark(precision, batch_size,imgsz, display_every=100, warm_up=10,repeat=10):

    print('\n=======================================================')
    print(f'Benchmark results for precision: {precision}, batch size: {batch_size}')
    print('=======================================================\n')
    
   
    # LOAD 만 해서 inference 진행
    model_name="yolov5s"
    trt_compiled_model_dir = f'{model_name}_saved_model_trt_saved_models/{model_name}_saved_model_fp32_64'
    # trt_compiled_model_dir = build_tensorrt_engine(precision, batch_size, dataset)

    walltime_start = time.time()
    m_load_time = time.time()
    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    load_time = time.time()-m_load_time

    pred_labels = []
    # actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every

    image_shape = (imgsz, imgsz,3)
    # image_shape = (3, imgsz, imgsz)
    data_shape = (batch_size,) + image_shape
    
    for i in range(repeat):
        dataset_s_time = time.time()
        img = np.random.uniform(-1, 1 , size=data_shape).astype("float32")
        d_load_time = time.time()-dataset_s_time

        if i==0:
            for w in range(warm_up):
                _ = model_trt(x=img);
                
        start_time = time.time()
        trt_results = model_trt(x=img);
        inference_time=time.time() - start_time

        if i ==0:
            first_iter_time = inference_time
        else:
            iter_times.append(inference_time)
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
    results.loc['total_inference_time']     = [np.sum(iter_times)*1000]
    results.loc['first_inference_time']     = [first_iter_time * 1000]
    results.loc['next_inference_time_mean'] = [np.median(iter_times[1:]) * 1000]
    results.loc['next_inference_time_mean'] = [np.mean(iter_times[1:]) * 1000]
    results.loc['images_per_sec_mean']      = [np.mean(batch_size / iter_times)]
    results.loc['model_load_time']          = [load_time*1000]
    results.loc['dataset_load_time']        = [d_load_time*1000]
    results.loc['wall_time']                = [(time.time() - walltime_start)*1000]
   
    return results, iter_times


if __name__ == "__main__":

    import argparse

    results = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='yolov5s',type=str)
    parser.add_argument('--batch_list',default=[1,2,4,8,16,32,64], type=list)
    parser.add_argument('--imgsize',default=640, type=int)
    parser.add_argument('--precision',default='fp32', type=str)
    args = parser.parse_args()
    model = args.model 
    batch_list = args.batch_list
    precision = args.precision
    imgsize = args.imgsize

    if results is None:
        results = pd.DataFrame()


    for batch_size in batch_list:
        opt = {'batch_size': batch_size}
        iter_ds = pd.DataFrame()
        
        print(f'{model}-{batch_size} start')
        
        res, it = trt_predict_benchmark(precision, batch_size,imgsize)
        # results = res        
        results = pd.concat([results, res], axis=1)
        print(results)
    results.to_csv(f'{model}_{batch_size}.csv')


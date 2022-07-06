import tensorflow as tf
import tensorflow.neuron as tfn
import warnings
import time
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import tensorflow.neuron as tfn
import os
import shutil
import json
import numpy as np
import time
import pandas as pd

model_name = model_type = 'bert-base-uncased'
inf1_model_dir = f'{model_type}_inf1_saved_models'
saved_model_dir = f'{model_type}_saved_model'

#benchmark batch 128 neuron model
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
compiled_batch_sizes = [1, 2, 4, 8, 16, 32, 64]

for compiled_batch in compiled_batch_sizes:
    iter_ds = pd.DataFrame()
    results = pd.DataFrame()
    walltime_start = time.time()
    
    compiled_model_dir = f'{model_type}_batch_{compiled_batch}_inf1'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)

    load_start = time.time()
    loaded_model = tf.keras.models.load_model(inf1_compiled_model_dir)
    load_time = time.time() - load_start
    
    
    result_list = pd.DataFrame()
    for batch_size in batch_sizes:
        first_iter_time = 0
        counter = 0
        iter_times = []
        seq_length = 128
        dtype = "int32"
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        for i in range(10):
            start_time = time.time()
            outputs = loaded_model(inputs)
            
            if counter ==0:
                first_iter_time = time.time() - start_time
            else:
                iter_times.append(time.time() - start_time)
                
            counter+=1
            
        iter_times = np.array(iter_times)
        results = pd.DataFrame(columns = [f'inf1_tf2_{model_type}_{compiled_batch}'])
        results.loc['batch_size']              = [batch_size]
        results.loc['accuracy']                = [0]
        results.loc['first_prediction_time']   = [first_iter_time * 1000]
        results.loc['next_inference_time_mean'] = [np.mean(iter_times) * 1000]
        results.loc['next_inference_time_median'] = [np.median(iter_times) * 1000]
        results.loc['load_time']               = [load_time * 1000]
        results.loc['wall_time']               = [(time.time() - walltime_start) * 1000]
        print(results)    
        result_list = pd.concat([result_list, results], axis = 1)
    print(result_list)

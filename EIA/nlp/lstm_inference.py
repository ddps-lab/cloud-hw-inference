import tensorflow as tf
from ei_for_tf.python.predictor.ei_predictor import EIPredictor
import numpy as np
import pandas as pd
import requests
import time
import boto3
import argparse
import tensorflow as tf
from tkinter import Y

from tensorflow import keras 
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

print(tf.__version__) 

ei_client = boto3.client('elastic-inference',region_name='us-west-2')


def get_dataset(max_review_length,top_words=5000):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    return X_train,y_train,X_test,y_test

def model_train_save(saved_model_dir,max_review_length,embedding_vector_length,batchsize,top_words=5000):
    X_train,y_train,X_test,y_test = get_dataset(max_review_length)
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=batchsize)

    model.save(saved_model_dir) 
    print("Train and Save Done")


def load_model(saved_model_dir):
    load_model_time = time.time()
    model = tf.keras.models.load_model(saved_model_dir)
    load_model_time = time.time() - load_model_time
    print(f"{saved_model_dir} load time : ",load_model_time)
    
    return model 

# 테스트 데이터를 배치 단위로 제공
def load_test_batch(batch_size,max_review_length,num_words=5000):

  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
  
  X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                        value= 0,
                                                        padding = 'pre',
                                                        maxlen = max_review_length )
  
  test_batch = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

  return test_batch

def ei_inference(saved_model_dir, batch_size, max_review_length,accelerator_id):
    
    ei_size = ei_client.describe_accelerators()['acceleratorSet'][accelerator_id]['acceleratorType']

    print('\n=======================================================')
    print(f'Benchmark results for EI: {ei_size}, batch size: {batch_size}')
    print('=======================================================\n')
    

    display_every = 5000
    warm_up = 10
    display_threshold = display_every

    iter_times = []
    pred_labels = []
    actual_labels = []

    load_start=time.time()
    print(saved_model_dir)
    eia_model = EIPredictor(saved_model_dir, 
                                accelerator_id=accelerator_id)
    load_time = time.time()-load_start
    print("Make EIA model time : ",load_time*1000,"ms")

    load_dataset_time = time.time()
    test_batch = load_test_batch(batch_size,max_review_length)
    load_dataset_time = time.time() - load_dataset_time
    print("dataset_load_time : ",load_dataset_time*1000,"ms")


    ipname = list(eia_model.feed_tensors.keys())[0]
    print(ipname)
    resname = list(eia_model.fetch_tensors.keys())[0]
    print(resname)
    walltime_start = time.time()
    for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
        model_feed_dict={f'{ipname}': X_test_batch.numpy()}

        if i == 0:
            for _ in range(warm_up):
                _ = eia_model(model_feed_dict)

        # 배치 단위별 데이터셋 분류
        inference_time = time.time()
        y_pred_batch = eia_model(model_feed_dict)
        inference_time = time.time() - inference_time

        if i ==0:
            first_iter_time = inference_time
        else:
            iter_times.append(inference_time)
      
        # actual_labels.extend(label for label_list in y_test_batch.numpy() for label in label_list)
        # actual_labels.append(y_test_batch.numpy())
        actual_labels.extend(label for label in y_test_batch.numpy() )
        pred_labels.extend(list(np.argmax(y_pred_batch[f'{resname}'], axis=1)))

        # 디버깅
        if i*batch_size >= display_threshold:
            print(f'Images {i*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every

    acc_ei_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    iter_times = np.array(iter_times)
    # acc_ei_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    
    results = pd.DataFrame(columns = [f'EIA_{saved_model_dir}_{batch_size}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['accelerator']             = [ei_size]
    results.loc['user_batch_size']         = [batch_size]
#     results.loc['accuracy']                = [acc_ei_gpu]
    results.loc['prediction_time']         = [np.sum(iter_times)]
    results.loc['images_per_sec_mean']     = [np.mean(batch_size / iter_times)]
    results.loc['first_iteration_time']   = [first_iter_time * 1000]
    results.loc['average_iteration_time'] = [np.mean(iter_times[1:]) * 1000]
    results.loc['latency_median']          = [np.median(iter_times) * 1000]
    results.loc['latency_99th_percentile'] = [np.percentile(iter_times, q=99, interpolation="lower") * 1000]
    results.loc['load_time']               = [load_time*1000]
    results.loc['wall_time']               = [(time.time() - walltime_start)*1000]
    
    return results, iter_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='lstm' , type=str)
    parser.add_argument('--save',default=False , type=bool)
    parser.add_argument('--seq_length',default=128 , type=int)
    parser.add_argument('--batch_list',default=[1,2,4,8,16,32,64], type=list)
    parser.add_argument('--eia_acc_id',default=0,type=int) 

    save = parser.parse_args().save
    model = parser.parse_args().model
    max_review_length = parser.parse_args().seq_length
    embedding_vector_length = parser.parse_args().seq_length
    batch_list = parser.parse_args().batch_list
    eia_acc_id = parser.parse_args().eia_acc_id

    saved_model_dir = f'{model}_imdb'

    results = pd.DataFrame()
    ei_size = ei_client.describe_accelerators()['acceleratorSet'][eia_acc_id]['acceleratorType']

    col_name = lambda ei_acc_id: f'ei_{ei_client.describe_accelerators()["acceleratorSet"][ei_acc_id]["acceleratorType"]}_batch_size_{batch_size}'

    # ei_options = [{'ei_acc_id': 0}]
    for batch_size in batch_list:
        if save : 
            # training 시킨 batchsize로만 추후 inference 가능 
            model_train_save(saved_model_dir,max_review_length,embedding_vector_length,batch_size)
        
        opt = {'batch_size': batch_size}
        iter_ds = pd.DataFrame()
        
        print(f'{model}-{batch_size}-{ei_size}start')
        res, iter_times = ei_inference(saved_model_dir, int(batch_size),max_review_length,eia_acc_id)
        
        col_name = lambda opt: f'{model}_{batch_size}_{ei_size}'
        
        iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
        results = pd.concat([results, res], axis=1)
        print(results)

    results.to_csv(f'{model}_EIA_{ei_size}.csv')

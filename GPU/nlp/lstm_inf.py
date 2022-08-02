from tkinter import Y
import numpy as np
import time 
import argparse
import requests
import pandas as pd 

import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


print(tf.__version__)
np.random.seed(7)

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


def load_test_batch(batch_size,max_review_length,num_words=5000):

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
  
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                        value= 0,
                                                        padding = 'pre',
                                                        maxlen = max_review_length )
  
    test_batch = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    return test_batch


def inference(saved_model_dir,batchsize,max_review_length,repeat=10):
    load_time = time.time()
    model = load_model(saved_model_dir)
    load_time = time.time()-load_time

    display_every = 5000
    warm_up = 10
    display_threshold = display_every

    acc_list=[]
    iter_times = []


    load_dataset_time = time.time()
    test_batch = load_test_batch(batchsize,max_review_length)
    load_dataset_time = time.time() - load_dataset_time
    print("dataset_load_time : ",load_dataset_time*1000,"ms")

    walltime_start = time.time()
    for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
        if i == 0:
            for _ in range(warm_up):
                _ = model.evaluate(X_test_batch)

        # 배치 단위별 데이터셋 분류
        inference_time = time.time()
        scores =  model.evaluate(X_test_batch,y_test_batch,verbose=0)
        inference_time = time.time() - inference_time
        print(scores[1]*100)

        if i ==0:
            first_iter_time = inference_time
        else:
            iter_times.append(inference_time)
      
        # actual_labels.extend(label for label_list in y_test_batch.numpy() for label in label_list)
        acc_list.append(scores[1]*100)

        # 디버깅
        if i*batchsize >= display_threshold:
            print(f'Images {i*batchsize}/50000. Average i/s {np.mean(batchsize/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every
    
    acc = acc_list[len(acc_list)-1]
    iter_times = np.array(iter_times)
    # acc_ei_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    
    results = pd.DataFrame(columns = [f'{saved_model_dir}_{batchsize}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['user_batch_size']         = [batchsize]
    results.loc['accuracy']                = [acc]
    results.loc['prediction_time']         = [np.sum(iter_times)]
    results.loc['images_per_sec_mean']     = [np.mean(batchsize / iter_times)]
    results.loc['first_iteration_time']   = [first_iter_time * 1000]
    results.loc['average_iteration_time'] = [np.mean(iter_times[1:]) * 1000]
    results.loc['latency_median']          = [np.median(iter_times) * 1000]
    results.loc['latency_99th_percentile'] = [np.percentile(iter_times, q=99, interpolation="lower") * 1000]
    results.loc['load_time']               = [load_time*1000]
    results.loc['wall_time']               = [(time.time() - walltime_start)*1000]
    
    return results, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',default=False , type=bool)
    parser.add_argument('--model',default='lstm' , type=str)
    parser.add_argument('--seq_length',default=128 , type=int)
    parser.add_argument('--batchsize',default=8 , type=int)


    save = parser.parse_args().save
    model = parser.parse_args().model
    max_review_length = parser.parse_args().seq_length
    embedding_vector_length = parser.parse_args().seq_length
    batchsize = parser.parse_args().batchsize
    saved_model_dir = f'{model}_imdb'

    if save : 
        # training 시킨 batchsize로만 추후 inference 가능 
        model_train_save(saved_model_dir,max_review_length,embedding_vector_length,batchsize)
    
    res,acc = inference(saved_model_dir,batchsize,max_review_length)
    print(f'{model} inference info : {res}s , {acc}% accuracy')

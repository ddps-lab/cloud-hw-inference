from tkinter import Y
import numpy as np
import time 
import argparse
from random import randint

import tensorflow as tf 
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

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

def inference(saved_model_dir,batchsize,max_review_length,repeat=10):
    X_train,y_train,X_test,y_test = get_dataset(max_review_length)
    model = load_model(saved_model_dir)
    time_list=[]
    acc_list=[]
    for i in range(repeat):
        # random 으로 dataset에서 batchsize 만큼 가져오기
        value = randint(0, len(X_test)-batchsize)
        X_test_batch = X_test[value:value+batchsize]
        y_test_batch = y_test[value:value+batchsize]

        start_time = time.time()
        scores = model.evaluate(X_test_batch, y_test_batch, verbose=0)
        running_time = time.time() - start_time
        time_list.append(running_time)
        acc_list.append(scores[1]*100)
        # print(i+1,"th Accuracy: %.2f%%" % (scores[1]*100))
        # print(i+1,"th Inference time:",running_time )

    res = np.median(np.array(time_list[1:]))
    acc = acc_list[len(acc_list)-1]
    return res, acc

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

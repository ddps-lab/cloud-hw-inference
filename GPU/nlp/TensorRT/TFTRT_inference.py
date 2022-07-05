import time
import pandas as pd
import numpy as np
import requests
import argparse
from random import randint
from transformers import BertTokenizer

import tensorflow as tf

from tensorflow.python.saved_model import tag_constants, signature_constants

print(f"TensorFlow version: {tf.__version__}")


results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50',type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--load_batchsize',default=8 , type=int)
parser.add_argument('--precision',default='fp32', type=str)
parser.add_argument('--size',default=224,type=int)
args = parser.parse_args()
model = args.model 
batch_size = args.batchsize
model_batchsize=args.load_batchsize
precision = args.precision
size=args.size


def index_to_word(x_train, y_train, index_word,batchsize):
  data = []
  value = randint(0, len(x_train)-batchsize)
  for i in range(value,value+batchsize):
    data.append(' '.join(index_word.get(index-3,'') for index in x_train[i]))
  y_train = y_train[value:value+batchsize]
  return data,y_train

def get_dataset():
  num_words = 5000
  # return 값은 인덱스 정수 목록인 시퀀스 목록
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
  imdb_dict = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
  index_word = dict((value,key) for key, value in imdb_dict.items())

  return X_train, y_train, X_test, y_test , index_word

def slice_dataset(X_train, y_train,X_test,y_test,index_word,batchsize):
  x_train_words, train_label = index_to_word(X_train,y_train, index_word,batchsize) 
  x_test_words, test_label = index_to_word(X_test, y_test, index_word,batchsize) 
  return x_train_words, train_label,x_test_words, test_label

def tokenization(max_seq_length,sentences,labels,tokenizer):
    input_ids = []
    attention_masks = []

    for sent in sentences:
      encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'tf',     # Return pytorch tensors.
                      )
        
        # Add the encoded sentence to the list.    
      input_ids.append(encoded_dict['input_ids'])
        
      attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = np.concatenate(input_ids, axis=0)
    attention_masks = np.concatenate(attention_masks, axis=0)
    labels = np.array(labels)
    return input_ids, attention_masks, labels





def trt_predict_benchmark(precision, batch_size,max_seq_length, display_every=100, warm_up=10,repeat=10):

    print('\n=======================================================')
    print(f'Benchmark results for precision: {precision}, batch size: {batch_size}')
    print('=======================================================\n')
    
    X_train, y_train, X_test, y_test , index_word = get_dataset()
 
   
    # LOAD 만 해서 inference 진행
    model_name="bert_base"
    trt_compiled_model_dir = f'{model_name}_imdb_trt_saved_models/{model_name}_imdb_{precision}_{model_batchsize}'
    # trt_compiled_model_dir = build_tensorrt_engine(precision, batch_size, dataset)

    walltime_start = time.time()

    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    pred_labels = []
    actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every
    initial_time = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for i in range(repeat):
        x_train_words, train_label,x_test_words, test_label = slice_dataset(X_train, y_train,X_test,y_test,index_word,batch_size)

        input_ids, attention_masks, labels = tokenization(max_seq_length,x_test_words,test_label,tokenizer)   

        if i==0:
            for w in range(warm_up):
                _ = model_trt(attention_mask=attention_masks,input_ids=input_ids);
                
        start_time = time.time()
        trt_results = model_trt(attention_mask=attention_masks,input_ids=input_ids);
        iter_times.append(time.time() - start_time)
        
        # actual_labels.extend(label for label_list in labels.numpy() for label in label_list)
        actual_labels.extend(label for label in labels )
        
        # print(trt_results)
        pred_labels.extend(list(tf.argmax(trt_results['logits'], axis=1).numpy()))
        if (i)*batch_size >= display_threshold:
            print(f'Images {(i)*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every
    
    print(f'Wall time: {time.time() - walltime_start}')

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
    print(results.T)
   
    return results, iter_times


if results is None:
    results = pd.DataFrame()

max_seq_length = 128
res, it = trt_predict_benchmark(precision, batch_size,max_seq_length)
results = res

print(results)
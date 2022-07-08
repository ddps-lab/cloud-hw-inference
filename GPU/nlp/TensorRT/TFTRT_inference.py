import time
import pandas as pd
import numpy as np
import requests
import argparse

import tensorflow as tf

from tensorflow.python.saved_model import tag_constants, signature_constants

print(f"TensorFlow version: {tf.__version__}")



# def index_to_word(x_train, y_train, index_word,batchsize):
#   data = []
#   value = randint(0, len(x_train)-batchsize)
#   for i in range(value,value+batchsize):
#     data.append(' '.join(index_word.get(index-3,'') for index in x_train[i]))
#   y_train = y_train[value:value+batchsize]
#   return data,y_train

# def get_dataset():
#   num_words = 5000
#   # return 값은 인덱스 정수 목록인 시퀀스 목록
#   (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
#   imdb_dict = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
#   index_word = dict((value,key) for key, value in imdb_dict.items())

#   return X_train, y_train, X_test, y_test , index_word

# def slice_dataset(X_train, y_train,X_test,y_test,index_word,batchsize):
#   x_train_words, train_label = index_to_word(X_train,y_train, index_word,batchsize) 
#   x_test_words, test_label = index_to_word(X_test, y_test, index_word,batchsize) 
#   return x_train_words, train_label,x_test_words, test_label

# def tokenization(max_seq_length,sentences,labels,tokenizer):
#     input_ids = []
#     attention_masks = []

#     for sent in sentences:
#       encoded_dict = tokenizer.encode_plus(
#                             sent,                      # Sentence to encode.
#                             add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                             max_length = max_seq_length,           # Pad & truncate all sentences.
#                             pad_to_max_length = True,
#                             return_attention_mask = True,   # Construct attn. masks.
#                             return_tensors = 'tf',     # Return pytorch tensors.
#                       )
        
#         # Add the encoded sentence to the list.    
#       input_ids.append(encoded_dict['input_ids'])
        
#       attention_masks.append(encoded_dict['attention_mask'])

#     # Convert the lists into tensors.
#     input_ids = np.concatenate(input_ids, axis=0)
#     attention_masks = np.concatenate(attention_masks, axis=0)
#     labels = np.array(labels)
#     return input_ids, attention_masks, labels


def make_dataset(batch_size, seq_length):
    inputs = np.random.randint(0, 200, size=(batch_size, seq_length)).astype(np.int32)
    token_types = np.random.uniform(size=(batch_size, seq_length)).astype(np.int32)
    valid_length = np.asarray([seq_length] * batch_size).astype(np.int32)
        
    return inputs, token_types, valid_length


def trt_predict_benchmark(trt_compiled_model_dir,precision, batch_size,max_seq_length, display_every=100, warm_up=10,repeat=10):

    print('\n=======================================================')
    print(f'Benchmark results for precision: {precision}, batch size: {batch_size}')
    print('=======================================================\n')
     
   
    # LOAD 만 해서 inference 진행
    walltime_start = time.time()
    load_s_time = time.time()
    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    load_time = time.time()-load_s_time
    
    pred_labels = []
    actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every
    # initial_time = time.time()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for i in range(repeat):
        # x_train_words, train_label,x_test_words, test_label = slice_dataset(X_train, y_train,X_test,y_test,index_word,batch_size)

        # input_ids, attention_masks, labels = tokenization(max_seq_length,x_test_words,test_label,tokenizer)   
        dataset_s_time = time.time()
        input_ids, attention_masks, valid_length = make_dataset(batch_size, max_seq_length)
        d_load_time = time.time()-dataset_s_time

        if i==0:
            for w in range(warm_up):
                _ = model_trt(attention_mask=attention_masks,input_ids=input_ids);
                
        start_time = time.time()
        trt_results = model_trt(attention_mask=attention_masks,input_ids=input_ids);
        inference_time=time.time() - start_time
        if i ==0:
            first_iter_time = inference_time
        else:
            iter_times.append(inference_time)
        # actual_labels.extend(label for label_list in labels.numpy() for label in label_list)
        # actual_labels.extend(label for label in labels )
        
        # print(trt_results)
        pred_labels.extend(list(tf.argmax(trt_results['logits'], axis=1).numpy()))

        if (i)*batch_size >= display_threshold:
            print(f'Images {(i)*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every
    
    print(f'Wall time: {time.time() - walltime_start}')

    # acc_trt = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    iter_times = np.array(iter_times)
   
    results = pd.DataFrame(columns = [f'{trt_compiled_model_dir}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['batch_size']               = [batch_size]
    # results.loc['accuracy']                 = [acc_cpu]
    results.loc['total_inference_time']     = [np.sum(iter_times)*1000]
    results.loc['first_inference_time']     = [first_iter_time * 1000]
    results.loc['next_inference_time_mean'] = [np.median(iter_times[1:]) * 1000]
    results.loc['next_inference_time_mean'] = [np.mean(iter_times[1:]) * 1000]
    results.loc['images_per_sec_mean']      = [np.mean(batch_size / iter_times)]
    results.loc['model_load_time']          = [load_time*1000]
    results.loc['dataset_load_time']        = [d_load_time*1000]
    results.loc['wall_time']                = [(time.time() - walltime_start)*1000]
    print(results.T)
   
    return results, iter_times


if __name__ == "__main__":
  import argparse

  results = None
  parser = argparse.ArgumentParser()
  parser.add_argument('--model',default='bert_base',type=str)
  parser.add_argument('--batch_list',default=[1,2,4,8,16,32,64], type=list)
  parser.add_argument('--engine_batchsize',default=64 , type=int)
  parser.add_argument('--max_seq_length',default=128 , type=int)
  parser.add_argument('--precision',default='fp32', type=str)
  
  args = parser.parse_args()
  model = args.model 
  batch_list = args.batch_list
  engine_batchsize = args.engine_batchsize
  precision = args.precision
  max_seq_length = args.max_seq_length

  trt_compiled_model_dir = f'{model}_trt_saved_models/{model}_{precision}_{engine_batchsize}'

  results = pd.DataFrame()

  for batch_size in batch_list:
    opt = {'batch_size': batch_size}
    iter_ds = pd.DataFrame()
    print(f'{model}-{batch_size} start')
    res, iter_times = trt_predict_benchmark(trt_compiled_model_dir,precision, batch_size,max_seq_length)
    col_name = lambda opt: f'{model}_{batch_size}'
        
    iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
    results = pd.concat([results, res], axis=1)
    print(results)
results.to_csv(f'{model}.csv')

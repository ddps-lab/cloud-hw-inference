import shutil
import numpy as np
import argparse
from random import randint
from transformers import BertTokenizer

import tensorflow as tf
from functools import partial

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To adapt!
print(f"TensorFlow version: {tf.__version__}")

import argparse

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50',type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--precision',default='fp32', type=str)
args = parser.parse_args()
model = args.model 
batch_size = args.batchsize
precision = args.precision



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




####tensorRT compile

def input_fn(batchsize,max_length):
    for _ in range(100):
        yield np.ones(shape=[batchsize, max_length], dtype=np.int32), \
            np.ones(shape=[batchsize, max_length], dtype=np.int32), \
            # np.ones(shape=[batchsize, max_length], dtype=np.int32)


def build_tensorrt_engine(model,precision, batch_size, max_length):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    print(f"TensorRT version: {trt.trt_utils._pywrap_py_utils.get_linked_tensorrt_version()}")

    input_saved_model = f'{model}'
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision.upper(),
                                                                   max_workspace_size_bytes=(1<<32),
                                                                   maximum_cached_engines=100)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model,
                                        conversion_params=conversion_params)
    
    converter.convert()
        
    trt_compiled_model_dir = f'{model}_trt_saved_models/{model}_{precision}_{batch_size}'
    shutil.rmtree(trt_compiled_model_dir, ignore_errors=True)

    converter.build(input_fn=partial(input_fn, batch_size, max_length))
    converter.save(output_saved_model_dir=trt_compiled_model_dir)

    print(f'\nOptimized for {precision} and batch size {batch_size}, directory:{trt_compiled_model_dir}\n')
    return trt_compiled_model_dir


saved_model_dir = "bert_base_imdb"
batchsize=1
max_seq_length = 128 

X_train, y_train, X_test, y_test , index_word = get_dataset()
x_train_words, train_label,x_test_words, test_label = slice_dataset(X_train, y_train,X_test,y_test,index_word,batchsize)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
input_ids, attention_masks, labels = tokenization(max_seq_length,x_test_words,test_label,tokenizer)

model = tf.keras.models.load_model(f"{saved_model_dir}")
inference_func = model.signatures["serving_default"]

trt_compiled_model_dir = build_tensorrt_engine(saved_model_dir,precision, batch_size, max_seq_length)

print("================================================")
print(trt_compiled_model_dir)
print("================================================")



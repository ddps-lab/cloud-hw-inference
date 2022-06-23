import time
import tensorflow as tf
from transformers import TFBertForSequenceClassification ,BertTokenizer
import numpy as np
import argparse
from random import randint

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))


def model_save(saved_model_dir,input_ids,attention_masks):
  load_model_time = time.time()
  model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["acc"])
  model.summary()
  load_model_time = time.time() - load_model_time
  print(f"{saved_model_dir} load time :",load_model_time)

  model._saved_model_inputs_spec = None
  inp = {"input_ids": input_ids, "attention_mask": attention_masks}
  model._set_save_spec(inp)
  model.save(f"{saved_model_dir}")


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


def inference(saved_model_dir,tokenizer,batchsize,repeat=10):
  X_train, y_train, X_test, y_test , index_word = get_dataset()

  load_model_time = time.time()
  model = tf.keras.models.load_model(f"{saved_model_dir}")
  inference_func = model.signatures["serving_default"]
  load_model_time = time.time() - load_model_time 
  print(f"{saved_model_dir} load time : ",load_model_time)

  time_list=[]
  for i in range(repeat):
    x_train_words, train_label,x_test_words, test_label = slice_dataset(X_train, y_train,X_test,y_test,index_word,batchsize)
    input_ids, attention_masks, labels = tokenization(max_seq_length,x_test_words,test_label,tokenizer)

    start_time = time.time()
    inference_func(input_ids=input_ids,attention_mask=attention_masks)
    running_time = time.time() - start_time
    time_list.append(running_time)
    # print(i+1,"th Inference time:",running_time )
  res = np.median(np.array(time_list[1:]))

  return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',default=False , type=bool)
    parser.add_argument('--model',default='bert_base' , type=str)
    parser.add_argument('--seq_length',default=128 , type=int)
    parser.add_argument('--batchsize',default=8 , type=int)


    save = parser.parse_args().save
    model = parser.parse_args().model
    max_seq_length = parser.parse_args().seq_length
    embedding_vector_length = parser.parse_args().seq_length
    batchsize = parser.parse_args().batchsize
    saved_model_dir = f'{model}_imdb'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if save : 
        # training 시킨 batchsize로만 추후 inference 가능 
        X_train, y_train, X_test, y_test , index_word = get_dataset()
        x_train_words, train_label,x_test_words, test_label = slice_dataset(X_train, y_train,X_test,y_test,index_word,batchsize)
        input_ids, attention_masks, labels = tokenization(max_seq_length,x_test_words,test_label,tokenizer)
        model_save(saved_model_dir,input_ids,attention_masks)
    

    res = inference(saved_model_dir,tokenizer,batchsize)
    print(f'{model} inference : {res*1000} ms ')

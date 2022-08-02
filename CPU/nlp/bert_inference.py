import time
import pandas as pd 
import numpy as np
import argparse
import tensorflow as tf
from transformers import TFBertForSequenceClassification 

def model_save(saved_model_dir,input_ids,attention_masks):
  load_model_time = time.time()
  model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,output_attentions=True)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["acc"])
  model.summary()
  load_model_time = time.time() - load_model_time
  print(f"{saved_model_dir} save time :",load_model_time)

  model._saved_model_inputs_spec = None
  inp = {"input_ids": input_ids, "attention_mask": attention_masks}
  model._set_save_spec(inp)
  model.save(f"{saved_model_dir}")



# fake dataset 
def make_dataset(batch_size, seq_length):
    inputs = np.random.randint(0, 200, size=(batch_size, seq_length)).astype(np.int64)
    token_types = np.random.uniform(size=(batch_size, seq_length)).astype(np.int64)
    valid_length = np.asarray([seq_length] * batch_size).astype(np.int64)
        
    return inputs, token_types, valid_length

# imdb dataset 
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


def inference(saved_model_dir,batch_size,max_seq_length,repeat=10,warm_up=3):
    # real dataset 
  #X_train, y_train, X_test, y_test , index_word = get_dataset()
    walltime_start=time.time()
    load_model_time = time.time()
    model = tf.keras.models.load_model(f"{saved_model_dir}")
    inference_func = model.signatures["serving_default"]
    load_time = time.time() - load_model_time 
    print(f"{saved_model_dir} load time : ",load_model_time)

    iter_times=[]
    for i in range(repeat):
        # real dataset 
        # x_train_words, train_label,x_test_words, test_label = slice_dataset(X_train, y_train,X_test,y_test,index_word,batchsize)
        # input_ids, attention_masks, labels = tokenization(max_seq_length,x_test_words,test_label,tokenizer)

        # fake dataset 
        make_dataset_time = time.time()
        input_ids, attention_masks, valid_length = make_dataset(batch_size, max_seq_length)
        d_load_time = time.time()-make_dataset_time

        if i==0:
            for _ in range(warm_up):
                            _ =  inference_func(input_ids=input_ids,attention_mask=attention_masks)


        start_time = time.time()
        inference_func(input_ids=input_ids,attention_mask=attention_masks)
        inference_time = time.time() - start_time

        if i ==0:
            first_iter_time = inference_time
        else:
            iter_times.append(inference_time)

    iter_times = np.array(iter_times)
    results = pd.DataFrame(columns = [f'CPU_{saved_model_dir}'])
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

    return results, iter_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',default=False , type=bool)
    parser.add_argument('--model',default='bert_base' , type=str)
    parser.add_argument('--seq_length',default=128 , type=int)
    parser.add_argument('--batch_list',default=[1,2,4,8,16,32,64], type=list)


    save = parser.parse_args().save
    model = parser.parse_args().model
    max_seq_length = parser.parse_args().seq_length
    embedding_vector_length = parser.parse_args().seq_length
    batch_list = parser.parse_args().batch_list

    results = pd.DataFrame()

    for batch_size in batch_list:
        opt = {'batch_size': batch_size}
        iter_ds = pd.DataFrame()
        
        if save and batch_size==1 : 
            input_ids, token_types, valid_length = make_dataset(batch_size, max_seq_length)
            model_save(model,input_ids,token_types)
    
        print(f'{model}-{batch_size} start')
        res, iter_times = inference(model,batch_size,max_seq_length)
        col_name = lambda opt: f'{model}_{batch_size}'
        
        iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
        results = pd.concat([results, res], axis=1)
        print(results)
    results.to_csv(f'{model}.csv')

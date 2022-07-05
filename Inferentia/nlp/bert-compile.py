from transformers import pipeline
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import tensorflow.neuron as tfn
import os

model_name = model_type = 'bert-base-uncased'

model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["acc"])
original_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=1, num_cores=1, use_static_weights=False):
    print(f'-----------batch size: {batch_size}, num cores: {num_cores}----------')
    print('Compiling...')
    
    compiled_model_dir = f'{model_type}_batch_{batch_size}_inf1_cores_{num_cores}'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
    shutil.rmtree(inf1_compiled_model_dir, ignore_errors=True)

    seq_length = 128
    dtype = "float32"
    inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
    
    start_time = time.time()
    compiled_model = tfn.trace(model, inputs, verbose=1, compiler_args = ['--neuroncore-pipeline-cores', str(core)])
    compiled_res = compiled_model.save(inf1_compiled_model_dir)
    print(f'Compile time: {time.time() - start_time}')
    
    compile_success = False
#     perc_on_inf = compiled_res['OnNeuronRatio'] * 100
#     if perc_on_inf > 50:
#         compile_success = True
            
    print(inf1_compiled_model_dir)
    print(compiled_res)
    print('----------- Done! ----------- \n')
    
    return compile_success                                                                   
                                                                    
                                                                    
inf1_model_dir = f'{model_type}_inf1_saved_models'
saved_model_dir = f'{model_type}_saved_model'


# testing batch size
batch_list = [1,2,4,8,16,32,64]
num_of_cores = [1, 2, 3, 4]
for batch in batch_list:
    for core in num_of_cores:
        print('batch size:', batch,'core nums', core,'compile start')
        compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=batch, num_cores=core)

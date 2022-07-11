import sys
# get_ipython().system('{sys.executable} -m pip install pillow matplotlib pycocotools==2.0.2 --force --extra-index-url=https://pip.repos.neuron.amazonaws.com')

# get_ipython().system('curl -LO http://images.cocodataset.org/zips/val2017.zip')
# get_ipython().system('curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
# get_ipython().system('unzip -q val2017.zip')
# get_ipython().system('unzip annotations_trainval2017.zip')

# get_ipython().system('ls')

import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import load_model

yolo_model = './yolo_v5_coco_saved_model'

import tensorflow as tf
import tensorflow.neuron as tfn
import os
import time
from tensorflow.keras.models import load_model
import numpy as np
import shutil

yolo_model = './yolo_v5_coco_saved_model'

model_type = 'yolo_v5_coco'

def no_fuse_condition(op):
    return op.name.startswith('Preprocessor') or op.name.startswith('Postprocessor')


def compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=1, use_static_weights=False):
    
    compiled_model_dir = f'{model_type}_batch_{batch_size}'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
    shutil.rmtree(inf1_compiled_model_dir, ignore_errors=True)
    
    model = load_model(yolo_model)
    
    example_input = np.zeros([batch_size,640,640,3], dtype='float32')
    start_time = time.time()
    compiled_model = tfn.trace(model,example_input) 
    compiled_res = compiled_model.save(inf1_compiled_model_dir)
    print(f'Compile time: {time.time() - start_time}')

    print(inf1_compiled_model_dir)
    print(compiled_res)
    print('----------- Done! ----------- \n')


# In[28]:


inf1_model_dir = f'{model_type}_inf1_saved_models'
saved_model_dir = f'{model_type}_saved_model'


# testing batch size
batch_list = [1,2,4,8,16,32,64]
num_of_cores = [1]
for batch in batch_list:
    print('batch size:', batch, 'compile start')
    compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=batch)





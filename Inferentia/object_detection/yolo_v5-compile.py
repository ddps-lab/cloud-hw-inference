#!/usr/bin/env python
# coding: utf-8

# # Evaluate YOLO v3 on Inferentia
# ## Note: this tutorial runs on tensorflow-neuron 1.x only

# ## Introduction
# This tutorial walks through compiling and evaluating YOLO v3 model on Inferentia using the AWS Neuron SDK.
# 
# 
# In this tutorial we provide two main sections:
# 
# 1. Download Dataset and Generate Pretrained SavedModel
# 
# 2. Compile the YOLO v3 model.
# 
# 3. Deploy the same compiled model.
# 
# Before running the following verify this Jupyter notebook is running “conda_aws_neuron_tensorflow_p36” kernel. You can select the Kernel from the “Kernel -> Change Kernel” option on the top of this Jupyter notebook page.
# 
# Instructions of how to setup Neuron Tensorflow environment and run the tutorial as a Jupyter notebook are available in the Tutorial main page [Tensorflow-YOLO_v3 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/yolo_v3_demo/yolo_v3_demo.html)

# ## Prerequisites
# 

# This demo requires the following pip packages:
# 
# `pillow matplotlib pycocotools`
# 

# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install pillow matplotlib pycocotools==2.0.2 --force --extra-index-url=https://pip.repos.neuron.amazonaws.com')
    


# ## Part 1:  Download Dataset and Generate Pretrained SavedModel
# ### Download COCO 2017 validation dataset
# 
# We start by downloading the COCO validation dataset, which we will use to validate our model. The COCO 2017 dataset is widely used for object-detection, segmentation and image captioning.

# In[4]:


get_ipython().system('curl -LO http://images.cocodataset.org/zips/val2017.zip')
get_ipython().system('curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
get_ipython().system('unzip -q val2017.zip')
get_ipython().system('unzip annotations_trainval2017.zip')


# In[5]:


get_ipython().system('ls')


# 
# ## Generate YOLO v3 tensorflow SavedModel (pretrained on COCO 2017 dataset)
# 
# Script yolo_v3_coco_saved_model.py will generate a tensorflow SavedModel using pretrained weights from https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz.

# This tensorflow SavedModel can be loaded as a tensorflow predictor. When a JPEG format image is provided as input, the output result of the tensorflow predictor contains information for drawing bounding boxes and classification results.

# In[7]:


import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import load_model

yolo_model = './yolo_v5_coco_saved_model'

# launch predictor and run inference on an arbitrary image in the validation dataset
yolo_pred_cpu = load_model(yolo_model)
image_path = './val2017/000000581781.jpg'
with open(image_path, 'rb') as f:
    feeds = {'image': [f.read()]}
results = yolo_pred_cpu(feeds)

# load annotations to decode classification result
with open('./annotations/instances_val2017.json') as f:
    annotate_json = json.load(f)
label_info = {idx+1: cat['name'] for idx, cat in enumerate(annotate_json['categories'])}

# draw picture and bounding boxes
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Image.open(image_path).convert('RGB'))
wanted = results['scores'][0] > 0.1
for xyxy, label_no_bg in zip(results['boxes'][0][wanted], results['classes'][0][wanted]):
    xywh = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    rect = patches.Rectangle((xywh[0], xywh[1]), xywh[2], xywh[3], linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    rx = rx + rect.get_width() / 2.0
    ax.annotate(label_info[label_no_bg + 1], (rx, ry), color='w', backgroundcolor='g', fontsize=10,
                ha='center', va='center', bbox=dict(boxstyle='square,pad=0.01', fc='g', ec='none', alpha=0.5))
plt.show()


# ## Part 2:  Compile the Pretrained SavedModel for Neuron
# 
# We make use of the Python compilation API `tfn.saved_model.compile` that is available in `tensorflow-neuron<2`. For the purpose of reducing Neuron runtime overhead, it is necessary to make use of arguments `no_fuse_ops` and `minimum_segment_size`.
# Compiled model is saved in ./yolo_v3_coco_saved_model_neuron.

# In[27]:


import tensorflow as tf
import tensorflow.neuron as tfn
import os
import time
from tensorflow.keras.models import load_model
import numpy as np

yolo_model = './yolo_v5_coco_saved_model'

model_type = 'yolo_v5_coco'

def no_fuse_condition(op):
    return op.name.startswith('Preprocessor') or op.name.startswith('Postprocessor')


def compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=1, num_cores=1, use_static_weights=False):
    
    compiled_model_dir = f'{model_type}_batch_{batch_size}_inf1_cores_{num_cores}'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
    shutil.rmtree(inf1_compiled_model_dir, ignore_errors=True)
    
    compiler_args = ['--verbose','1', '--neuroncore-pipeline-cores', str(num_cores)]
    model = load_model(yolo_model)
    
    example_input = np.zeros([batch_size,640,640,3], dtype='float32')
    start_time = time.time()
    compiled_model = tfn.trace(model,example_input) 
    compiled_res = compiled_model.save(inf1_model_dir)
    print(f'Compile time: {time.time() - start_time}')

#     result = tfn.saved_model.compile(
#         saved_model_dir, compiled_model_dir,
#         # to enforce trivial compilable subgraphs to run on CPU
#     #     no_fuse_ops=no_fuse_ops,
#         minimum_segment_size=100,
#         batch_size=batch_size,
#         dynamic_batch_size=True,
#         compiler_args = compiler_args
#     )
#     print(result)

    print(inf1_compiled_model_dir)
    print(compiled_res)
    print('----------- Done! ----------- \n')


# In[28]:


inf1_model_dir = f'{model_type}_inf1_saved_models'
saved_model_dir = f'{model_type}_saved_model'


# testing batch size
batch_list = [1]
num_of_cores = [1]
for batch in batch_list:
    for core in num_of_cores:
        print('batch size:', batch,'core nums', core,'compile start')
        compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=batch, num_cores=core)


# In[ ]:





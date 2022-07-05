import tensorflow as tf
import tensorflow.neuron as tfn
import warnings
import time
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import tensorflow.neuron as tfn
import os
import shutil
import json
import numpy as np
import time

inf1_model_dir = f'{model_type}_inf1_saved_models'
saved_model_dir = f'{model_type}_saved_model'
model_name = model_type = 'bert-base-uncased'

#benchmark batch 128 neuron model
neuron_b128_times = []
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
compiled_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
for compiled_batch in compiled_batch_sizes:
    compiled_model_dir = f'{model_type}_batch_{compiled_batch_size}_inf1'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
    loaded_model = tf.keras.models.load_model(inf1_compiled_model_dir)
    for batch_size in batch_sizes:
        seq_length = 128
        dtype = "int32"
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        for i in range(10):
            start = time.time()
            outputs = loaded_model(inputs)
            end = time.time()
            neuron_b128_times.append(end - start)

        neuron_b128_times = sorted(neuron_b128_times)

        print(f"Average throughput for batch 128 neuron model is {128/(sum(neuron_b128_times)/len(neuron_b128_times))} sentences/s.")
        print(f"Peak throughput for batch 128 neuron model is {128/min(neuron_b128_times)} sentences/s.")
        print()


        print(f"50th percentile latency for batch 128 neuron model is {neuron_b128_times[int(1000*.5)] * 1000} ms.")
        print(f"90th percentile latency for batch 128 neuron model is {neuron_b128_times[int(1000*.9)] * 1000} ms.")
        print(f"95th percentile latency for bacth 128 neuron model is {neuron_b128_times[int(1000*.95)] * 1000} ms.")
        print(f"99th percentile latency for batch 128 neuron model is {neuron_b128_times[int(1000*.99)] * 1000} ms.")
        print()

from transformers import pipeline
import tensorflow as tf
import tensorflow.neuron as tfn
import warnings
import time

class TFBertForSequenceClassificationDictIO(tf.keras.Model):
    def __init__(self, model_wrapped):
        super().__init__()
        self.model_wrapped = model_wrapped
        self.aws_neuron_function = model_wrapped.aws_neuron_function
    def call(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        logits = self.model_wrapped([input_ids, attention_mask])
        return [logits]


reloaded_model = tf.keras.models.load_model('./distilbert_b128')
rewrapped_model = TFBertForSequenceClassificationDictIO(model_wrapped_traced)

#now you can reinsert our reloaded model back into our pipeline
neuron_pipe.model = rewrapped_model
neuron_pipe.model.config = pipe.model.config


warnings.warn("NEURONCORE_GROUP_SIZES is being deprecated, if your application is using NEURONCORE_GROUP_SIZES please \
see https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/deprecation.html#announcing-end-of-support-for-neuroncore-group-sizes \
for more details.", DeprecationWarning)


string_inputs = [
    'I love to eat pizza!',
    'I am sorry. I really want to like it, but I just can not stand sushi.',
    'I really do not want to type out 128 strings to create batch 128 data.',
    'Ah! Multiplying this list by 32 would be a great solution!',
]
string_inputs = string_inputs * 32

#warmup inf
neuron_pipe(string_inputs)
#benchmark batch 128 neuron model
neuron_b128_times = []
for i in range(1000):
    start = time.time()
    outputs = neuron_pipe(string_inputs)
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

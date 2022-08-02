import tensorflow as tf
import tf2onnx
from transformers import TFBertForSequenceClassification ,BertTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bucket', type=str)
parser.add_argument('--model', default='bert', type=str)
args = parser.parse_args()
model_name = args.model
bucket_name = args.bucket


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["acc"])
model.summary()

input_spec = (
    tf.TensorSpec((1,  512), tf.int32, name="input_ids"),
    tf.TensorSpec((1,  512), tf.int32, name="token_type_ids"),
    tf.TensorSpec((1,  512), tf.int32, name="attention_mask")
)

_, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13, output_path="tf_bert.onnx")
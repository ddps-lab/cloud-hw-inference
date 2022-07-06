import shutil
import numpy as np
import argparse

import tensorflow as tf
from functools import partial

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To adapt!
print(f"TensorFlow version: {tf.__version__}")

####tensorRT compile

def input_fn(batch_size,imgsz):
    image_shape = (imgsz, imgsz,3)
    # image_shape = (3, imgsz, imgsz)
    data_shape = (batch_size,) + image_shape
    
    for _ in range(100):
        img = np.random.uniform(-1, 1 , size=data_shape).astype("float32")
        yield (img,)


def build_tensorrt_engine(model,precision, batch_size, imgsz):
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

    converter.build(input_fn=partial(input_fn, batch_size, imgsz))

    converter.save(output_saved_model_dir=trt_compiled_model_dir)

    print(f'\nOptimized for {precision} and batch size {batch_size}, directory:{trt_compiled_model_dir}\n')
    return trt_compiled_model_dir




if __name__ == "__main__":

    import argparse

    results = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='yolov5s',type=str)
    parser.add_argument('--batchsize',default=64 , type=int)
    parser.add_argument('--imgsize',default=640, type=int)
    parser.add_argument('--precision',default='fp32', type=str)
    args = parser.parse_args()
    model = args.model 
    batch_size = args.batchsize
    precision = args.precision

    saved_model_dir = f"{model}_saved_model"
    batchsize=1
    imgsz = 640
    # model = tf.keras.models.load_model(f"{saved_model_dir}")
    # inference_func = model.signatures["serving_default"]

    trt_compiled_model_dir = build_tensorrt_engine(saved_model_dir,precision, batch_size, imgsz)

    print("================================================")
    print(trt_compiled_model_dir)
    print("================================================")


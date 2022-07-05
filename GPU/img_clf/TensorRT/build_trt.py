import os
from functools import partial

import tensorflow as tf
from tensorflow.keras.applications import ( 
    vgg16,
    resnet50,
    inception_v3,
    mobilenet_v2,
    xception,
)
models = {
    'xception':xception,
    'vgg16':vgg16,
    'resnet50':resnet50,
    'inception_v3':inception_v3,
    'mobilenet_v2':mobilenet_v2
}


def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                  'image/class/text': tf.io.FixedLenFeature([], tf.string, '')}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    label_text = tf.cast(obj['image/class/text'], tf.string)   
    return imgdata, label, label_text

def val_preprocessing(record):
    imgdata, label, label_text = deserialize_image_record(record)
    label -= 1
    image = tf.io.decode_jpeg(imgdata, channels=3, 
                              fancy_upscaling=False, 
                              dct_method='INTEGER_FAST')

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)

    scale = tf.cond(tf.greater(height, width),
                  lambda: side / width,
                  lambda: side / height)
    
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    
    image = models[model].preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batchsize):
    data_dir = '/workspace/datasets/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batchsize)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset

def calibrate_fn(n_calib, batchsize, dataset):
    for i, (calib_image, _, _) in enumerate(dataset):
        if i > n_calib // batchsize:
            break
        yield (calib_image,)

def build_fn(dataset):
    for i, (build_image, _, _) in enumerate(dataset):
        if i > 1:
            break
        yield (build_image,)

def build_FP_tensorrt_engine(model,precision, batchsize, dataset,num_engines):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    if precision == 'FP32':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP32,
                                                        max_workspace_size_bytes=8000000000,maximum_cached_engines=num_engines)
    elif precision == 'FP16':                                                 
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP16,
                                                        max_workspace_size_bytes=8000000000,maximum_cached_engines=num_engines)
    
    elif precision=='INT8':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.INT8, 
                                                        max_workspace_size_bytes=8000000000, 
                                                        use_calibration=True ,maximum_cached_engines=num_engines )
    

    saved_model_dir = f"../{model}"
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir,
                                        conversion_params=conversion_params)
    
    if precision=='INT8':
        #converter.convert(calibration_input_fn=calibration_input_fn)
        n_calib=50
        converter.convert(calibration_input_fn=partial(calibrate_fn, n_calib, batchsize, 
                                                       dataset.shuffle(buffer_size=n_calib, reshuffle_each_iteration=True)))
    else:
        converter.convert()
    

    trt_compiled_model_dir = f'{model}_{precision}_{batchsize}'
    print("TensorEngine Build")
    converter.build(input_fn=partial(build_fn, dataset))
    print("tensorRT_SAVED")
    converter.save(output_saved_model_dir=trt_compiled_model_dir)
    print(f'\nOptimized for {precision} and engine batch size {batchsize}, directory:{trt_compiled_model_dir}\n')

    return trt_compiled_model_dir


if __name__ == "__main__":
    import argparse

    results = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=64,type=int)
    parser.add_argument('--precision',default='FP32',type=str)
    parser.add_argument('--num_engines',default=100,type=int)

    args = parser.parse_args()
    model = args.model
    # engine batchsize
    batchsize = args.batchsize
    precision = args.precision
    num_engines = args.num_engines

    dataset = get_dataset(batchsize)

    print("------TENSORRT BUILD ENGINE-----")
    trt_compiled_model_dir = build_FP_tensorrt_engine(model,precision, batchsize, dataset,num_engines)
    print("Done")

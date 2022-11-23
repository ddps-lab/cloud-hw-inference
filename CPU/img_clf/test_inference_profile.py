import os, time
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import (
    xception,
    vgg16,
    resnet50,
    inception_v3,
    mobilenet_v2,
)

models = {
    'xception': xception,
    'vgg16': vgg16,
    'resnet50': resnet50,
    'inception_v3': inception_v3,
    'mobilenet_v2': mobilenet_v2,
}


def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                   'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1)}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)
    return imgdata, label


def val_preprocessing(record):
    imgdata, label = deserialize_image_record(record)
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
    if "inception" in model or "xception" in model:
        image = tf.image.resize_with_crop_or_pad(image, 299, 299)
    else:
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    label = tf.cast(label, tf.int32)
    image = models[model].preprocess_input(image)
    image = tf.cast(image, tf.float32)
    return image, label


def get_dataset(batch_size, use_cache=False):
    data_dir = '/home/ubuntu/datasets/images-1000/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)

    return dataset


def inference(saved_model_name, batch_size):
    walltime_start = time.time()
    first_iter_time = 0
    iter_times = []
    pred_labels = []
    actual_labels = []
    total_datas = 50000
    display_every = 5000
    display_threshold = display_every
    warm_up = 10

    d_start_time = time.time()
    ds = get_dataset(batch_size)
    d_load_time = time.time() - d_start_time

    load_start = time.time()
    model = load_model(saved_model_name)
    load_time = time.time() - load_start

    counter = 0
    for batch, batch_labels in ds:

        tf.profiler.experimental.start('profile')
        start_time = time.time()
        yhat_np = model.predict(batch)
        inference_time = time.time() - start_time
        tf.profiler.experimental.stop()
        break

        if counter == 0:
            first_iter_time = inference_time
        else:
            iter_times.append(inference_time)

        actual_labels.extend(label for label_list in batch_labels for label in label_list)
        pred_labels.extend(list(np.argmax(yhat_np, axis=1)))

        if counter * batch_size >= display_threshold:
            print(
                f'Images {counter * batch_size}/{total_datas}. Average i/s {np.mean(batch_size / np.array(iter_times[-display_every:]))}')
            display_threshold += display_every

        counter += 1

    iter_times = np.array(iter_times)
    acc_cpu = np.sum(np.array(actual_labels) == np.array(pred_labels)) / len(actual_labels)

    results = pd.DataFrame(columns=[f'CPU_{saved_model_name}'])
    results.loc['batch_size'] = [batch_size]
    results.loc['accuracy'] = [acc_cpu]
    results.loc['total_inference_time'] = [np.sum(iter_times) * 1000]
    results.loc['first_inference_time'] = [first_iter_time * 1000]
    results.loc['next_inference_time_mean'] = [np.median(iter_times[1:]) * 1000]
    results.loc['next_inference_time_mean'] = [np.mean(iter_times[1:]) * 1000]
    results.loc['images_per_sec_mean'] = [np.mean(batch_size / iter_times)]
    results.loc['model_load_time'] = [load_time * 1000]
    results.loc['dataset_load_time'] = [d_load_time * 1000]
    results.loc['wall_time'] = [(time.time() - walltime_start) * 1000]

    return results, iter_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--batch', default=1, type=int)

    model = parser.parse_args().model
    saved_model = model + '_saved_model'
    batch_size = parser.parse_args().batch

    results = pd.DataFrame()

    opt = {'batch_size': batch_size}
    iter_ds = pd.DataFrame()

    print(f'{model}-{batch_size} start')
    res, iter_times = inference(saved_model, int(batch_size))
    col_name = lambda opt: f'{model}_{batch_size}'

    iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
    results = pd.concat([results, res], axis=1)
    print(results)
    results.to_csv(f'{model}.csv')

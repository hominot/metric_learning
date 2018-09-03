from tqdm import tqdm
import math
import requests
import tarfile
import tensorflow as tf
import os


def download(url, directory='/tmp/research', filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    wrote = 0
    with open(filepath, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print('ERROR, something went wrong')
    return filepath


def extract(filepath):
    extract_path = os.path.dirname(filepath)
    tar = tarfile.open(filepath, 'r')
    for item in tar:
        tar.extract(item, extract_path)


def create_image_dataset(file_paths, labels):
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [250, 250])
        return image_resized, label
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(_parse_function)
    return dataset

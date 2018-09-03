from util.dataset import download, extract, create_image_dataset
import os


def create_dataset():
    filepath = download('http://vis-www.cs.umass.edu/lfw/lfw.tgz')
    extract(filepath)
    label_map = {}
    labels = []
    image_files = []
    for subdir, dirs, files in os.walk('{}/lfw'.format(os.path.dirname(filepath))):
        for file in files:
            label = subdir.split('/')[-1]
            if label not in label_map:
                label_map[label] = len(label_map) + 1
            image_files.append(os.path.join(subdir, file))
            labels.append(label_map[label])

    return create_image_dataset(image_files, labels)

from tensorflow.keras.applications.resnet50 import ResNet50

from util.registry.model import Model


class Resnet50Model(Model):
    name = 'resnet50'

    def __init__(self, conf, extra_info):
        super(Resnet50Model, self).__init__(conf, extra_info)

        width = conf['image']['width'] if 'random_crop' not in conf['image'] else conf['image']['random_crop']['width']
        height = conf['image']['height'] if 'random_crop' not in conf['image'] else conf['image']['random_crop']['height']
        channel = conf['image']['channel']
        self.model = ResNet50(
            include_top=False, pooling='max', weights='imagenet', input_shape=(width, height, channel))

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from util.registry.model import Model


class InceptionModel(Model):
    name = 'inception'

    def __init__(self, conf, extra_info):
        super(InceptionModel, self).__init__(conf, extra_info)

        self.model = InceptionResNetV2(
            include_top=False, pooling='max', weights='imagenet')

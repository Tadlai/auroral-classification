from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2

from extractor import Extractor
import tensorflow as tf

class ResNet(Extractor):
    def __init__(self, size=224, **kwargs):
        super().__init__(**kwargs)
        self._name = "ResnetExtractor"
        self.size = size
        i = tf.keras.layers.Input([size, size, 3])
        x = tf.keras.applications.resnet50.preprocess_input(i)
        res = ResNet50(include_top=False,
                       weights="imagenet",
                       input_shape=(size, size, 3))
        x = res(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.encoder = tf.keras.Model(inputs=[i], outputs=[x])


class Inception(Extractor):
    def __init__(self, size=299, **kwargs):
        super().__init__(**kwargs)
        self._name = "InceptionExtractor"
        self.size = size
        i = tf.keras.layers.Input([size, size, 3])
        x = tf.image.resize(i,(299, 299))
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        inc = InceptionV3(include_top=False,
                          weights="imagenet",
                          input_shape=(299, 299, 3))
        x = inc(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.encoder = tf.keras.Model(inputs=[i], outputs=[x])


class MobileNet(Extractor):
    def __init__(self, size=224, **kwargs):
        super().__init__(**kwargs)
        self._name = "MobilenetExtractor"
        self.size = size
        i = tf.keras.layers.Input([size, size, 3])
        x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
        mob = MobileNetV2(include_top=False,
                          weights="imagenet",
                          input_shape=(size, size, 3))
        x = mob(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.encoder = tf.keras.Model(inputs=[i], outputs=[x])

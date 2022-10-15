import os
from datetime import datetime

from matplotlib import pyplot as plt

from extractor import Extractor
from loader import get_single_image
from simCLR_helpers import _dot_simililarity_dim1, _dot_simililarity_dim2, CustomAugment, get_negative_mask
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tqdm import tqdm
import tensorflow as tf
import numpy as np
# REFERENCE : https://github.com/google-research/simclr

class SimCLRExtractor(Extractor):
    def __init__(self, lr, batch_size, n_epochs, size=224, temperature=0.5,
                 transformations=None, model_path=None,hidden_1=256,hidden_2=128, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        if transformations is None: # WARNING does not work
            transformations = {'fliplr': False, 'flipud': True, 'crop': True, 'colorj': False, 'colord': False}
        self._name = "SimCLRExtractor"
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.decay_steps = 1000
        # self.lr_decayed_fn = tf.keras.experimental.CosineDecay(
        #     initial_learning_rate=lr, decay_steps=self.decay_steps)
        # self.optimizer = tf.keras.optimizers.SGD(self.lr_decayed_fn)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.size = size
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.temperature = temperature
        if model_path is None:
            self.projection = self.get_resnet_simclr(hidden_1, hidden_2)

        self.transformations = transformations
        # self.data_augmentation = Sequential([Lambda(CustomAugment())])
        self.data_augmentation = CustomAugment()
        # Mask to remove positive examples from the batch of negative samples
        self.negative_mask = get_negative_mask(self.batch_size)

        # self.encoder.summary()

    def get_resnet_simclr(self, hidden_1, hidden_2):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,
                                                    input_shape=(self.size, self.size, 3))
        base_model.trainable = True
        inputs = Input((self.size, self.size, 3))
        resnet_simclr = Sequential()
        resnet_simclr.add(inputs)
        resnet_simclr.add(base_model)
        resnet_simclr.add(GlobalAveragePooling2D())
        resnet_simclr.add(Dense(hidden_1))
        resnet_simclr.add(Activation("relu"))
        resnet_simclr.add(Dense(hidden_2))

        self.encoder = Sequential([inputs, base_model, GlobalAveragePooling2D()])
        # resnet_simclr.summary()

        return resnet_simclr

    def get_inception_simclr(self, hidden_1, hidden_2):
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None,
                                                       input_shape=(self.size, self.size, 3))
        base_model.trainable = True
        inputs = Input((self.size, self.size, 3))
        inception_simclr = Sequential()
        inception_simclr.add(inputs)
        inception_simclr.add(base_model)
        inception_simclr.add(GlobalAveragePooling2D())
        inception_simclr.add(Dense(hidden_1))
        inception_simclr.add(Activation("relu"))
        inception_simclr.add(Dense(hidden_2))

        self.encoder = Sequential([inputs, base_model, GlobalAveragePooling2D()])

        return inception_simclr

    def compute_loss(self, xis, xjs):
        zis = self.projection(xis)
        zjs = self.projection(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = _dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (self.batch_size, 1))
        l_pos /= self.temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = _dot_simililarity_dim2(positives, negatives)

            labels = tf.zeros(self.batch_size, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, self.negative_mask)
            l_neg = tf.reshape(l_neg, (self.batch_size, -1))
            l_neg /= self.temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss += self.criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * self.batch_size)
        return loss

    def compute_lossv2(self, xis, xjs):
        zis = self.projection(xis, training=True)
        zjs = self.projection(xjs, training=True)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=-1)
        zjs = tf.math.l2_normalize(zjs, axis=-1)

        labels = tf.one_hot(tf.range(self.batch_size), self.batch_size * 2)
        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

        logits_aa = tf.matmul(zis, zis, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * 1e9
        logits_bb = tf.matmul(zjs, zjs, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * 1e9
        logits_ab = tf.matmul(zis, zjs, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(zjs, zis, transpose_b=True) / self.temperature

        loss_a = tf.compat.v1.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], 1))

        loss_b = tf.compat.v1.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], 1))

        loss = loss_a + loss_b
        return loss

    def train_step(self, xis, xjs):
        with tf.GradientTape() as tape:
            loss = self.compute_lossv2(xis, xjs)

        gradients = tape.gradient(loss, self.projection.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.projection.trainable_variables))

        return loss

    def validation_step(self, xis, xjs):
        loss = self.compute_lossv2(xis, xjs)
        return loss

    def train_tf(self, train_ds, valid_ds, n_samples, valid_len, input_data_path):
        plt.ion()
        plt.show()
        self.input_data_path = input_data_path

        train_ds = train_ds.map(get_single_image)
        valid_ds = valid_ds.map(get_single_image)

        epoch_wise_loss = []
        epoch_wise_val_loss = []

        for epoch in tqdm(range(self.n_epochs)):
            step_wise_loss = []
            step_wise_val_loss = []

            # Training steps
            batch_count = 0
            for image_batch in train_ds:
                a, b = self.data_augmentation(image_batch)

                loss = self.train_step(a, b)
                step_wise_loss.append(loss)
                batch_count += 1
                print("batch: {} loss: {:.3f}".format(batch_count, loss))
                # plt.figure()
                # for i in range(len(a)):
                #     # plt.subplot(3, len(a), i + 1)
                #     # plt.imshow(image_batch[i, :, :, :])
                #     # plt.axis('off')
                #     plt.subplot(2, len(a), i + 1)
                #     plt.imshow(a[i, :, :, :])
                #     plt.axis('off')
                #     plt.subplot(2, len(a), len(a) + i + 1)
                #     plt.imshow(b[i, :, :, :])
                #     plt.axis('off')
                #
                # plt.subplots_adjust(bottom=0, top=1, wspace=0, hspace=0)
                # plt.draw()
                # plt.pause(0.001)

            # Validation steps
            for image_batch in valid_ds:
                a, b = self.data_augmentation(image_batch)
                loss = self.validation_step(a, b)
                step_wise_val_loss.append(loss)

            epoch_wise_loss.append(np.mean(step_wise_loss))
            epoch_wise_val_loss.append(np.mean(step_wise_val_loss))
            print("epoch: {} loss: {:.3f} val_loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss),
                                                                   np.mean(step_wise_val_loss)))

        plt.ioff()
        self.save_custom()
        return epoch_wise_loss, epoch_wise_val_loss

    def save_custom(self):
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        self.encoder.save("/home/vincent/models/" + date_time + "_enc")
        print("Model saved at location: " + "/home/vincent/models/" + date_time)

    def save_machine(self, dir):
        super().save_machine(dir)
        self.encoder.save(os.path.join(dir, "encoder"))


if __name__ == "__main__":
    s = SimCLRExtractor(lr=0.001, batch_size=128, n_epochs=1, size=224)
    r = s.get_resnet_simclr(10, 20)
    r.save("temp_model")
    res = tf.keras.models.load_model("temp_model")

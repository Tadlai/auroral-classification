import os
from loader import duplicate_img

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from datetime import datetime
from custom_callbacks import DisplayResultsCallback
from extractor import Extractor


class AutoencoderModel(Model):
    def __init__(self, n_filters, n_layers, size=256, kernel_size=(3, 3), encoded_size = 128,full_conv=False):
        super().__init__()
        list_layers_enc = [layers.Input(shape=(size, size, 3))]
        for i in range(n_layers):
            list_layers_enc.append(layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same', strides=2))
            list_layers_enc.append(layers.BatchNormalization())
        if not full_conv:
            list_layers_enc.append(layers.Flatten())
            list_layers_enc.append(layers.Dense(encoded_size, activation='relu'))
            list_layers_dec = [layers.Dense(((size // (2 ** n_layers)) ** 2) * n_filters),
                               layers.Reshape(target_shape=(size//(2 ** n_layers), size//(2 ** n_layers), n_filters))]
        else:
            list_layers_enc.append(layers.Flatten())
            list_layers_dec = [layers.Reshape(target_shape=(size//(2 ** n_layers), size//(2 ** n_layers), n_filters))]

        for i in range(n_layers):
            list_layers_dec.append(layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, activation='relu', padding='same'))
            list_layers_enc.append(layers.BatchNormalization())
        list_layers_dec.append(layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))
        self.encoder = tf.keras.Sequential(list_layers_enc)
        self.decoder = tf.keras.Sequential(list_layers_dec)


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNNAutoencoder(Extractor):
    def __init__(self, n_filters, n_layers, lr, batch_size, n_epochs, input_size=256, kernel_size=(3, 3), encoded_size=128,
                 full_conv=True,model_path=None, **kwargs):
        super(CNNAutoencoder, self).__init__(model_path=model_path, **kwargs)
        self._name = "AutoencoderExtractor"
        if model_path is None:
            self.model = AutoencoderModel(n_filters, n_layers, input_size, kernel_size, encoded_size,full_conv)
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.encoded_size = encoded_size
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs


    def save_custom(self):
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        self.model.encoder.save("/home/vincent/models/" + date_time + "_enc")
        self.model.decoder.save("/home/vincent/models/" + date_time + "_dec")
        print("Model saved at location: " + "/home/vincent/models/" + date_time)

    def train_tf(self, train_ds, valid_ds, n_samples, valid_len, input_data_path):
        self.input_data_path = input_data_path
        train_ds = train_ds.map(duplicate_img)
        valid_ds = valid_ds.map(duplicate_img)

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="auto",
            restore_best_weights=True,
        )
        disp_callback = DisplayResultsCallback(valid_ds)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss=losses.MeanSquaredError(), metrics=['accuracy'])

        history = self.model.fit(train_ds,
                                 epochs=self.n_epochs,
                                 shuffle=True,
                                 validation_data=valid_ds,
                                 callbacks=[callback,disp_callback],
                                 verbose=1)  # custom callback not excluded for now

        self.save_custom()

        return history.history["loss"], history.history["val_loss"]

    def save_machine(self, dir):
        super().save_machine(dir)
        self.encoder.save(os.path.join(dir, "encoder"))


if __name__ == "__main__":
    c = CNNAutoencoder(n_filters=2, n_layers=2, lr=0.1, batch_size=32, n_epochs=2)
    print(c.get_params())

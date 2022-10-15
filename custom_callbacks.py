import tensorflow.keras
import matplotlib.pyplot as plt


class DisplayResultsCallback(tensorflow.keras.callbacks.Callback):
    """
    Tensorflow custom callback function allowing to display the results of the autoencoder at each epoch.
    """
    def __init__(self, valid_ds):
        super(DisplayResultsCallback, self).__init__()
        self.disp_ds = valid_ds.unbatch().take(6).batch(batch_size=6)
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.disp_ds)
        orig = self.disp_ds.as_numpy_iterator()
        plt.figure()
        i = 0
        for imgs1,imgs2 in orig:
            for img in imgs1:
                i += 1
                plt.subplot(2, 6, i)
                plt.imshow(img)
        i = 0
        for pred in preds:
            i += 1
            plt.subplot(2, 6, i+6)
            plt.imshow(pred)

        plt.suptitle("Epoch "+str(epoch))
        plt.draw()
        plt.pause(0.001)

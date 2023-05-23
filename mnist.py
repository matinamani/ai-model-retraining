import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models, optimizers
from keras.datasets import mnist
from keras.utils import to_categorical

sns.set_theme()


class Mnist_Model:
    def __init__(self, epochs, batch_size, learning_rate, hidden_layers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers

    def run(self):
        self.__load_data()
        self.__preprocess()
        self.__build_network()
        self.__validate_training()

    def train(self):
        self.__load_data()
        self.__preprocess()
        self.__build_network()
        self.network.fit(
            self.train_imgs,
            self.train_lbls,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def __load_data(self):
        (self.train_imgs, self.train_lbls), (
            self.test_imgs,
            self.test_lbls,
        ) = mnist.load_data()

    def __preprocess(self):
        self.train_imgs = self.train_imgs.reshape((60000, 28 * 28))
        self.train_imgs = self.train_imgs.astype("float32") / 255
        self.test_imgs = self.test_imgs.reshape((10000, 28 * 28))
        self.test_imgs = self.test_imgs.astype("float32") / 255
        self.train_lbls = to_categorical(self.train_lbls)
        self.test_lbls = to_categorical(self.test_lbls)

        self.validation_imgs = self.train_imgs[:10000]
        self.validation_lbls = self.train_lbls[:10000]
        self.partial_train_imgs = self.train_imgs[10000:]
        self.partial_train_lbls = self.train_lbls[10000:]

    def __build_network(self):
        self.network = models.Sequential()

        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            if i == 0:
                self.network.add(
                    layers.Dense(
                        layer["neurons"],
                        activation=layer["activation"],
                        input_shape=(28 * 28,),
                    ),
                )
            else:
                self.network.add(
                    layers.Dense(
                        layer["neurons"],
                        activation=layer["activation"],
                    ),
                )

        self.network.compile(
            optimizer=optimizers.legacy.RMSprop(
                learning_rate=self.learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def __validate_training(self):
        self.history = self.network.fit(
            self.partial_train_imgs,
            self.partial_train_lbls,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.validation_imgs, self.validation_lbls),
        )

    def plot(self):
        epochs = range(1, len(self.history.history["loss"]) + 1)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.axes[0].plot(
            epochs, self.history.history["loss"], "bo", label="Training Loss"
        )

        self.axes[0].plot(
            epochs, self.history.history["val_loss"], label="Validation Loss"
        )

        self.axes[1].plot(
            epochs,
            self.history.history["accuracy"],
            "bo",
            label="Training Accuracy",
        )

        self.axes[1].plot(
            epochs,
            self.history.history["val_accuracy"],
            label="Validation Accuracy",
        )

        self.fig.suptitle(
            f"batch size={self.batch_size} - LR={self.learning_rate}",
            fontsize=22,
        )

        self.axes[0].set_xlabel("epochs")
        self.axes[1].set_xlabel("epochs")
        self.axes[0].set_title("Loss")
        self.axes[1].set_title("Accuracy")

        self.axes[0].legend()
        self.axes[1].legend()

        self.axes[0].locator_params(nbins=self.epochs)
        self.axes[1].locator_params(nbins=self.epochs)
        file_name = (
            f"mnist-fig;bs:{self.batch_size}_lr:{self.learning_rate}.png"
        )
        self.fig.savefig(file_name)
        # files.download(file_name)

    def eval(self):
        self.test_loss, self.test_acc = self.network.evaluate(
            self.test_imgs, self.test_lbls
        )

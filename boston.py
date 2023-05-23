import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import layers, models, optimizers
from keras.datasets import boston_housing

sns.set_theme()


class Boston_Model:
    def __init__(
        self,
        epochs,
        learning_rate,
        hidden_layers,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers

    def run(self, k):
        self.__load_data()
        self.__preprocess()
        fold_length = len(self.train_data) // k
        for i in range(k):
            print(f"Processing fold #{i}")
            (
                validation_data,
                validation_targets,
                partial_train_data,
                partial_train_targets,
            ) = self.__build_folds(i, fold_length)

            self.__build_network()

            history = self.__validate_training(
                validation_data,
                validation_targets,
                partial_train_data,
                partial_train_targets,
            )

            self.mae_data.append(history.history["val_mae"])

    def train(self):
        self.__load_data()
        self.__preprocess()
        self.__build_network()
        self.network.fit(
            self.train_data,
            self.test_data,
            epochs=self.epochs,
            batch_size=1,
        )

    def __build_folds(self, i, fold_length):
        validation_data = self.train_data[
            i * fold_length : (i + 1) * fold_length
        ]
        validation_targets = self.train_targets[
            i * fold_length : (i + 1) * fold_length
        ]
        partial_train_data = np.concatenate(
            [
                self.train_data[: i * fold_length],
                self.train_data[(i + 1) * fold_length :],
            ],
            axis=0,
        )
        partial_train_targets = np.concatenate(
            [
                self.train_targets[: i * fold_length],
                self.train_targets[(i + 1) * fold_length :],
            ],
            axis=0,
        )

        return (
            validation_data,
            validation_targets,
            partial_train_data,
            partial_train_targets,
        )

    def __load_data(self):
        (self.train_data, self.train_targets), (
            self.test_data,
            self.test_targets,
        ) = boston_housing.load_data()

    def __preprocess(self):
        self.mae_data = []
        self.train_data -= self.train_data.mean(axis=0)
        self.train_data /= self.train_data.std(axis=0)
        self.test_data -= self.train_data.mean(axis=0)
        self.test_data /= self.train_data.std(axis=0)

    def __build_network(self):
        self.network = models.Sequential()

        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            if i == 0:
                self.network.add(
                    layers.Dense(
                        layer["neurons"],
                        activation=layer["activation"],
                        input_shape=(self.train_data.shape[1],),
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
            optimizer=optimizers.RMSprop(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

    def __validate_training(
        self,
        validation_data,
        validation_targets,
        partial_train_data,
        partial_train_targets,
    ):
        return self.network.fit(
            partial_train_data,
            partial_train_targets,
            epochs=self.epochs,
            batch_size=1,
            validation_data=(validation_data, validation_targets),
            verbose=0,
        )

    def plot(self):
        avg_mae = [
            np.mean([x[i] for x in self.mae_data]) for i in range(self.epochs)
        ]

        self.fig, self.ax = plt.subplots()
        self.ax.plot(range(1, self.epochs + 1), avg_mae)
        self.ax.set_xlabel("epochs")
        self.ax.set_ylabel("MAE")

        self.ax.set_title("Validation MAE", fontsize=18)

        file_name = f"boston-fig;layers:[{','.join(map(str, [layer['neurons'] for layer in self.hidden_layers]))}].png"
        self.fig.savefig(file_name)

    def eval(self):
        self.test_mse, self.test_mae = self.network.evaluate(
            self.test_data, self.test_targets
        )

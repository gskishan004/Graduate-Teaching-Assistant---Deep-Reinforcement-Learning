import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

print(f"TensorFlow version = {tf.__version__}\n")

class Model:
    model = Sequential()
    def __init__(self):
        history_length = 1
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape=(96, 96, history_length)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size = 3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size = 3,  activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size = 3,  activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, activation='softmax'))

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer= tf.keras.optimizers.Adam(0.01),
        metrics=['accuracy'])


    def load_model(self):
        self = keras.models.load_model("./models/my_model_dag3")
        print(self.summary())
        return self

    def train(self, X_train, y_train, X_valid, y_valid):
        history = self.model.fit(X_train, y_train,
        batch_size=64,
        epochs=30,
        verbose=1,
        validation_data=(X_valid, y_valid))

    def save(self, file_name):
        self.save(f'./models/{file_name}')

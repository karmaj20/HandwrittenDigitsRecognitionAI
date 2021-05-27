import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
# use tensorboard command : tensorboard --logdir logs/

class NeuralNetwork:
    X_train, y_train, X_test, y_test = (None, None, None, None)
    model = None
    img_width = 28
    img_height = 28


    def __init__(self, choice):
        print(choice)
        self.loadDatabase()
        self.model = self.trainModel(choice)
        self.showAccuracy()
        self.checkModel()
        self.showLoss()
        self.showAccuracy()

    def loadDatabase(self):
        mnist = tf.keras.datasets.mnist
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = (np.expand_dims(self.X_train, axis=-1) / 255.).astype(np.float32)
        self.X_test = (np.expand_dims(self.X_test, axis=-1) / 255.).astype(np.float32)

    def createDenseModel(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
        model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/Dense', histogram_freq=1)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model, tb_callback

    def createConvolutionalModel(self):
        input_shape = (self.img_height, self.img_width, 1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation=tf.nn.relu, input_shape=input_shape))
        model.add(Conv2D(64, (3,3), activation=tf.nn.relu))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation=tf.nn.softmax))

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/Conv', histogram_freq=1)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model, tb_callback

    def trainModel(self, choice):
        if choice == 'C':
            model, tb_callback = self.createConvolutionalModel()
            model.fit(self.X_train, self.y_train, batch_size=128, epochs=10, verbose=1, validation_data=(self.X_test, self.y_test), callbacks=[tb_callback])
        elif choice == 'D':
            model, tb_callback = self.createDenseModel()
            model.fit(self.X_train, self.y_train, batch_size=128, epochs=10, verbose=1, validation_data=(self.X_test, self.y_test), callbacks=[tb_callback])
        else:
            model, tb_callback = self.createDenseModel()
            model.fit(self.X_train, self.y_train, batch_size=128, epochs=10, verbose=1,
                      validation_data=(self.X_test, self.y_test), callbacks=[tb_callback])

        return model

    def showAccuracy(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(accuracy)
        print(loss)

    def checkModel(self):
        self.model.summary()

    def predictModel(self, image):
        predictions = self.model.predict(image)
        return predictions

    def showLoss(self):
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss plot')
        plt.legend(['train', 'test'], loc='upper right')
        plt.tight_layout()
        plt.show()

    def showAccuracy(self):
        plt.plot(self.model.history.history['accuracy'])
        plt.plot(self.model.history.history['val_accuracy'])
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Accuracy plot')
        plt.legend(['train', 'test'], loc='lower right')
        plt.tight_layout()
        plt.show()

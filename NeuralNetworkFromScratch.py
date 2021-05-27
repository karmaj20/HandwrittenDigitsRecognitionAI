from Model import *
from Optimizer import *

class NeuralNetworkFromScratch:
    def __init__(self):
        self.loadDatabase()
        self.model = self.trainModel()

    def loadDatabase(self):
        self.X, self.y, self.X_test, self.y_test = NeuralNetworkFromScratch.create_data_mnist('mnist_png')

        # Shuffle the training dataset
        self.keys = np.array(range(self.X.shape[0]))
        np.random.shuffle(self.keys)
        self.X = self.X[self.keys]
        self.y = self.y[self.keys]

        # scale and reshape samples
        self.X = (self.X.reshape(self.X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
        self.X_test = (self.X_test.reshape(self.X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    def createDenseModel(self):
        # instantiate the model
        model = Model()

        # add layers
        model.add(Layer_Dense(self.X.shape[1], 128))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(128, 128))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(128, 10))
        model.add(Activation_Softmax())

        return model

    def trainModel(self):
        model = self.createDenseModel()

        # set loss, optimizer and accuracy objects
        model.set(
            loss=Loss_CategoricalCrossentropy(),
            optimizer=Optimizer_Adam(decay=1e-4),
            accuracy=Accuracy_Categorical()
        )

        # Finalize the model
        model.finalize()

        # Train the model
        model.train(self.X, self.y, validation_data=(self.X_test, self.y_test), epochs=1, batch_size=128, print_every=100)

        # # Save the model
        # model.save('digits.model')

        # Evaluate the model
        model.evaluate(self.X_test, self.y_test)

        return model

    @staticmethod
    def load_mnist_dataset(dataset, path):
        # scan all the directories and create a list of labels
        labels = os.listdir(os.path.join(path, dataset))

        # Create lists for samples and labels
        X = []
        y = []

        # For each label folder
        for label in labels:
            # and for each image in given folder
            for file in os.listdir(os.path.join(path, dataset, label)):
                # Read the image
                image = cv2.imread(os.path.join(
                    path, dataset, label, file
                ), cv2.IMREAD_UNCHANGED)

                # And append it and a label to the lists
                X.append(image)
                y.append(label)

        # Convert the data to proper numpy arrays and return
        return np.array(X), np.array(y).astype('uint8')

    @staticmethod
    # MNIST dataset (train + test)
    def create_data_mnist(path):
        # Load both sets separately
        X, y = NeuralNetworkFromScratch.load_mnist_dataset('train', path)
        X_test, y_test = NeuralNetworkFromScratch.load_mnist_dataset('test', path)

        # And return all the data
        return X, y, X_test, y_test

    def predictModel(self, image):
        predictions = self.model.predict(image)
        return predictions
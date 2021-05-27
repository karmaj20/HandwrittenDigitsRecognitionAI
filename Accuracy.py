from Imports import *

# Common accuray class
class Accuracy:

    # calculates an accuracy
    # given predicitons and ground truth values
    def calculate(self, predictions, y):

        # get comparison results
        comparisons = self.compare(predictions, y)

        # calculate an accuracy
        accuracy = np.mean(comparisons)

        # add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # return accuracy
        return accuracy

    # calculates accumulated acuracy
    def calculate_accumulated(self):

        # calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # return the data and regularization losses
        return accuracy

    # reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):

    # No initalization is needed
    def init(self, y):
        pass

    # compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
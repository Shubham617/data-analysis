import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from Data import Data


"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


class LinearRegression:
    def __init__(self, flag, alpha=1e-2):
        self.flag = flag        # flag can only be either: 'GradientDescent' or 'Analytic'
        self.alpha = alpha      # regularization coefficient
        self.w = np.zeros(2)    # initializing the weights

    def fit(self, X, Y, epochs=10000):
        """
        :param X: input features , shape: (N,2)
        :param Y: target, shape: (N,)
        :return: None

        IMP: Make use of self.w to ensure changes in the weight are reflected globally.
            We will be making use of get_params and set_params to check for correctness
        """
        if self.flag == 'GradientDescent':
            step = 0.001
            for t in range(epochs):
                # WRITE the required CODE HERE to update the parameters using gradient descent
                # make use of self.loss_grad() function
                self.w =  self.w - step*self.loss_grad(self.w, X, Y)
                #print(self.loss(self.w, X, Y))
                if t%100 == 0:
                    print("Epoch: {} :: loss: {}".format(t, self.loss(self.w, X, Y)))

        elif self.flag == 'Analytic':
            pass
            # Remember: matrix inverse in NOT equal to matrix**(-1)
            # WRITE the required CODE HERE to update the weights using closed form solution
            temp1 = np.dot(X.T, X) + 2 * 0.01 * np.eye(X.shape[1])
            temp2 = np.linalg.inv(temp1).dot(X.T)
            self.w = np.dot(temp2, Y)

        else:
            raise ValueError('flag can only be either: ''GradientDescent'' or ''Analytic''')

    def predict(self, X):
        """
        :param X: inputs, shape: (N,2)
        :return: predictions, shape: (N,)



        Return your predictions
        """
        # WRITE the required CODE HERE and return the computed values

        return np.dot(X, self.w)

    def loss(self, w, X, Y):
        """
        :param W: weights, shape: (2,)
        :param X: input, shape: (N,2)
        :param Y: target, shape: (N,)
        :return: scalar loss value

        Compute the loss loss. (Function will be tested only for gradient descent)
        """
        # WRITE the required CODE HERE and return the computed values
        l = np.dot(X, w) - Y
        l = np.dot(l.T, l)*0.5

        return l + 0.01*np.dot(w.T, w)

    def loss_grad(self, w, X, y):
        """
        :param W: weights, shape: (2,)
        :param X: input, shape: (N,2)
        :param Y: target, shape: (N,)
        :return: vector of shape: (2,) containing gradients for each weight

        Compute the gradient of the loss. (Function will be tested only for gradient descent)
        """
        # WRITE the required CODE HERE and return the computed values
        temp1 = np.dot(X.T, X) + 2*0.01*np.eye(X.shape[1])
        temp2 = np.dot(X.T, y)

        return np.dot(temp1, w) - temp2

    def get_params(self):
        """
        ********* DO NOT EDIT ************
        :return: the current parameters value
        """
        return self.w

    def set_params(self, w):
        """
        ********* DO NOT EDIT ************
        This function Will be used to set the values of weights externally
        :param w:
        :return: None
        """
        self.w = w
        return 0



if __name__ == '__main__':
    # Get data
    data = Data()
    X, y = data.get_linear_regression_data()

    # Linear regression with gradient descent
    model = LinearRegression(flag='GradientDescent')
    model.fit(X,y)
    y_grad = model.predict(X)
    w_grad = model.get_params()

    # Analytic Linear regression
    model = LinearRegression(flag='Analytic')
    model.fit(X,y)
    y_exact = model.predict(X)
    w_exact = model.get_params()

    # Compare the resultant parameters
    diff = norm(w_exact - w_grad)
    print(diff, w_exact, w_grad)

    # Plot the results
    plt.plot(X[:, 0], y, 'o-', label='true')
    plt.plot(X[:, 0], y_exact, '-', label='pinv')
    plt.plot(X[:, 0], y_grad, '-.', label='grad')
    plt.legend()
    plt.savefig('figures/Q1.png')
    plt.close()

from scipy.special import gamma
import numpy as np
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

class Posterior:
    def __init__(self, limes, cherries, a=2, b=2):
        self.a = a
        self.b = b
        self.limes = limes          # shape: (N,)
        self.cherries = cherries    # scalar int
        self.N = np.shape(self.limes)[0]

    def get_MAP(self):
        """
        compute MAP estimate
        :return: MAP estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        result = np.zeros(self.N)
        for i in range(self.N):
            N = i
            map = (self.a - 1)/(N +self.a + self.b - 2)

            result[i] = 1 - map

        return result

    def get_finite(self):
        """
        compute posterior with finite hypotheses
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        result = np.zeros(self.N)
        import math

        #math.pow(9,9)
        for i in range(self.N):
            N = i
            val1 = 0.2*math.pow(0.25,N+1)+0.4*math.pow(0.5,N+1)+0.2*math.pow(0.75,N+1)+0.1

            val2 = 0.2*math.pow(0.25,N+1)+2*0.4*math.pow(0.5,N+1)+0.2*math.pow(0.75,N+1)+0.1
            val3 = 0.75*0.2*math.pow(0.25,N)+0.25*0.2*math.pow(0.75,N)
            val4 = val2 + val3
            print(val1/(0.2*math.pow(0.25,N)+0.4*math.pow(0.5,N)+0.2*math.pow(0.75,N)+0.1))
            result[i] = val1/val4

        print(result)
        return result

    def get_infinite(self):
        """
        compute posterior with beta prior
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        result = np.zeros(self.N)

        for i in range(self.N):
            N = i
            lime = (gamma(self.a + self.b)/gamma(self.b)) * (gamma(N + self.b + 1)/(gamma(self.a + self.b + N +1)))
            cherry = (gamma(self.a + self.b)/(gamma(self.a)*gamma(self.b))) * ((gamma(self.a + 1)*gamma(N+self.b))/gamma(self.a + self.b + N +1))
            alpha = 1 /(lime + cherry)
            result[i] = alpha * lime

        return result

if __name__ == '__main__':
    # Get data
    data = Data()
    limes, cherries = data.get_bayesian_data()

    # Create class instance
    posterior = Posterior(limes=limes, cherries=cherries)

    # PLot the results
    plt.plot(limes, posterior.get_MAP(), label='MAP')
    plt.plot(limes, posterior.get_finite(), label='5 Hypotheses')
    plt.plot(limes, posterior.get_infinite(), label='Bayesian with Beta Prior')
    plt.legend()
    plt.savefig('figures/Q4.png')

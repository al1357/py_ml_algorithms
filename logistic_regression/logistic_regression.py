import numpy as np
import matplotlib.pyplot as plt
from numpy import float64
import math
"""
To do:
 - implement regularization
 - implement testing
"""
class logistic_regression:
    # x (m, n)
    X = None
    X_normal = None
    X_train = None
    X_test = None
    # labels (m, 1)
    Y = None
    Y_train = None
    Y_test = None
    
    # bias scalar
    b = None
    # thetas (1, n)
    theta = None
    # learing rate
    alpha = None
    iter = 500
    # number of rows/training examples
    m = 0
    m_train = 0
    m_test = 0
    # number of features without bias
    n = None
    
    norm = False
    mean = None
    std_dev = None
    
    def __init__(self, X, Y, normalize=False, normalize_columns=[], alpha_in=0.03, iter_in=500, test_split=0.2):
        # Fill in defaults
        to_shuffle = np.concatenate((X, Y), axis=1)
        np.random.shuffle(to_shuffle)
        
        self.X = to_shuffle[:, :-1]
        self.Y = to_shuffle[:, -1].reshape(-1, 1)
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        
        self.theta = np.zeros((self.n, 1), dtype="float64")
        self.b = 1
        self.iter = iter_in
        
        # Normalize
        self.norm = normalize
        if(normalize and len(normalize_columns) > 0):
            self.normalize_columns = normalize_columns
        elif(normalize):
            self.normalize_columns = list(range(0, self.n))
            
        if(normalize):
            self.features_normalize()
        
        # Split
        self.m_test = math.floor(self.m * test_split)
        self.X_test = self.X_normal[0:self.m_test, :] if (normalize) else self.X[0:self.m_test, :]
        self.Y_test = self.Y[0:self.m_test, :]
        if(normalize):
            self.X_train = self.X_normal[self.m_test:, :]
        else:
            self.X_train = self.X[self.m_test:, :]
        self.Y_train = self.Y[self.m_test:, :]
        
    #end
    
    def features_normalize(self, x_in=None): 
        """Normalizes features: >>> (feature - mean) / standard deviation <<<
        """
        if(self.mean is None):
            self.mean = np.sum(self.X[:, self.normalize_columns], 0) / self.m
            
        if(self.std_dev is None):
            self.std_dev = np.std(self.X[:, self.normalize_columns], 0)
        
        x = self.X if(x_in is None) else x_in
        x_norm = (x[:, self.normalize_columns] - self.mean) / self.std_dev
        
        x_norm_all = np.copy(x)
        norm_col_i = 0
        
        for col_i in self.normalize_columns:
            x_norm_all[:, col_i] = x_norm[:, norm_col_i]
            norm_col_i += 1
        
        if(self.X_normal is None and x_in is None):
            self.X_normal = x_norm_all
        
        return x_norm_all
    #end
    
    def sigmoid(self, z):
        """Sigmoid function
        prediciton: sigmoid >= 0.5 result 1; sigmoid < 0.5 result 0;
        """
        return 1 / (1 + np.exp(-z))
    #end
    
    def compute_cost(self, x, y):
        """Computes cost function
            x(m, n), y(m, 1)
        """
        z = np.dot(x, self.theta) + self.b # (m, n)@(n, 1) = (m, 1)
        a = self.sigmoid(z) # (m, 1) > SIGMOID - converts prediction to 0-1 range
        m_local = x.shape[0]
        # log(a) is measure of an error; a=sigmoid(z) => 0<a<1 => -inf<log(a)<1
        # log(1) = 0 - the lowest possible outcome; 
        # for y=1 low error is when log(a) is close to 0, so when a -> 1; 
        # for y=0 low error is when log(1-a) is colose to 0 when a -> 0;
        # in log(a) if a < 1 then log(a) < 0 - hence it's multiplied by -y or -(1-y), to make the result positive;
        
        # np.log of very small number results in division by 0 error
        # https://stackoverflow.com/questions/46510929/mle-log-likelihood-for-logistic-regression-gives-divide-by-zero-error
        pred41 =  np.dot(-y.T, np.log(a)) # (1, m) x (m, 1)
        pred40 = np.dot((1. - y).T, np.log(1. - a))
        cost = ( pred41 - pred40) / m_local    # (1, 1) > COST; convert to scalar with .max()
        return cost #.max() # ?
    #end
    
    def computeDz(self):      
        """Compute partial derivative dz on the training set - how J changes with respect to W(?)
        """
        a = self.predict(self.x)
        # a prediction 0-1; self.y 0 or 1 label
        return a - self.y
    #end
    
    def computeGradient(self, dz):
        """Computes gradient on the training set
        """
        gradient = (np.dot(self.x, dz.T)) / self.m # (n, m)@(1, m).T = (n, 1)
        return gradient
    #end
  
    def train(self):
        """Runs gradient descent by iterating over gradient function.
        """
# =============================================================================
#         print("Initial cost: ",self.compute_cost(self.X_train, self.Y_train))
#         dz = self.computeDz()
#         print("Initial bias gradient: ",(np.sum(dz).max() / self.m))
#         print("Initial theta gradient: ",self.computeGradient(dz))
# =============================================================================
      
        for i in range(self.interations):
            dz = self.computeDz()
            self.theta = self.theta - (self.alpha * self.computeGradient(dz))
            self.b = self.b - (self.alpha * (np.sum(dz).max() / self.m))
            
        print("Final cost: ",self.computeCost(self.x, self.y))
        print("Final bias: ",self.b)
        print("Final weights: ",self.theta)
    #end
    
    def predict(self, x):
        p = self.sigmoid(np.dot(x, self.theta) + self.b)
        return p
    #end
# =============================================================================
#     
#     def getAccuracy(self, x, y):
#         p = self.predict(x)
#         pBinary = np.where(p>=0.5,1,0)
#         pyCompare = (pBinary == y)
#         accuracy = np.mean(pyCompare) * 100
#         print("Accuracy: ",accuracy)  
#     #end
#     
#     def predictTest(self):
#         student = np.array([[45], [85]], dtype="float64")
#         studentNorm = self.featuresNormalize(student, True)
#         print("Normalized student scores: ",studentNorm)
#         studentPrediction = self.predict(studentNorm)
#         print("Probability of 1: ",studentPrediction)
#     #end
# =============================================================================

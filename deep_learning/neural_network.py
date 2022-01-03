# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#from numpy import float64, int
import math
import performance_measure as pm
"""

"""
class neural_network:
    
    # n x m
    X = []
    X_cv = []
    X_test = []
    Y = []
    Y_cv = []
    Y_test = []
    
    # number of rows/training examples
    m = 0
    m_cv = 0
    m_test = 0
    # number of features
    n = 0
    network_structure = None
    # z cache
    cache_z = []
    cache_a = []
    cache_dw = {}
    cache_db = {}
    # error cache iteration-error
    cache_train_error = []
    cache_cv_error = []
    cache_iterations = []
    cache_mb_train_error = []
    cache_mb_cv_error = []
    cache_mb_iterations = []
    # learing rate
    weights = {}
    bias = {}
    buffer = {}
    
    regularization = True
    reg_lambda = 0.00001
    dropout = False
    dropout_keep_prob = 0.8
    
    X_min = 0
    X_max = 0
    
    performance_measure = None
    last_layer_af = "softmax"
    
    early_stopping = False
    mini_batch_size_n = -1 # 2^n
    iterations = 100
    max_iterations = 100000
    alpha = 0.1

    
    def __init__(self, network_structure, \
                     X, \
                     Y, \
                     sets_distribution=[0.8, 0.1, 0.1], \
                     load_parameters=False):
        """
            training/test sets dimensions are n x m
        """
        np.random.seed(3)
        self.network_structure = network_structure
        
        labels_count = Y.shape[0]
        to_shuffle = np.concatenate((X, Y), axis=0)
        np.random.shuffle((to_shuffle.T))

        self.X = to_shuffle[:-labels_count, :]
        self.Y = to_shuffle[-labels_count:, :]
        
        self.X_min = np.amin(self.X, axis=1).reshape(-1, 1)
        self.X_max = np.amax(self.X, axis=1).reshape(-1, 1)
        
        self.m = self.X.shape[1]
        self.n = self.X.shape[0]
        # split into cv and test
        if(sets_distribution[1] and sets_distribution[1] + sets_distribution[2] < 0.6):
            self.m_cv = math.ceil(self.m * sets_distribution[1])
            self.X_cv = self.X[:, :self.m_cv]
            self.Y_cv = self.Y[:, :self.m_cv]
            
        if(sets_distribution[2] and sets_distribution[1] + sets_distribution[2] < 0.6):
            self.m_test = math.ceil(self.m * sets_distribution[2])
            self.X_test = self.X[:, self.m_cv:(self.m_cv + self.m_test)]
            self.Y_test = self.Y[:, self.m_cv:(self.m_cv + self.m_test)]
        
        self.m = self.m - self.m_cv - self.m_test
        self.X = self.X[:, (self.m_cv + self.m_test):]
        self.Y = self.Y[:, (self.m_cv + self.m_test):]
        
        print("Training samples: ", self.X.shape)
        if(self.m_cv): print("CV samples: ", self.X_cv.shape)
        if(self.m_test): print("Test samples: ", self.X_test.shape)
        
        self.performance_measure = pm.performance_measure(self)
  
        # load saved parameters
        if load_parameters:
            self.load_parameters()
        else:   
            self.initialize_parameters("he")
        #end if
        
        #self.scale()
        self.standardize()
    #end
    
    def save_parameters(self):
        for l in self.weights:
            with open("parameters/weights_l"+str(l)+".txt", "w") as outfile:
                outfile.write("# Weights for layer {0}; shape {1}\n".format(l, self.weights[l].shape))
                np.savetxt(outfile, self.weights[l])
            #end with
        #end for
        
        with open("parameters/bias.txt", "w") as outfile:
            outfile.write("# Bias dict. length: {0}\n".format(len(self.bias))) 
            for i in self.bias:
                outfile.write("# Bias for layer {0}; shape {1}\n".format(i, self.bias[i].shape))
                np.savetxt(outfile, self.bias[i])
            #end for
        #end with
    #end
    
    def load_parameters(self):
        try:
            i = 1
            for nl in self.network_structure:
                self.weights[i] = np.loadtxt("parameters/weights_l"+str(i)+".txt")
                i = i + 1
            #end for
            bias = np.loadtxt("parameters/bias.txt")            
        except Exception as e:
            print("Weights/bias file read error. Initializing random parameters.")
            print("Error msg: "+str(e))
            if not bool(self.weights) or not bool(self.bias):    
                self.initialize_parameters("he")
        else:
            prev_layer_size = 0
            i=1
            for nl in self.network_structure:
                self.bias[i] = np.array(bias[prev_layer_size:prev_layer_size + nl]).reshape(nl, 1)
                prev_layer_size += nl
                i += 1
            #end for
        #end else
    #end
        
    def initialize_parameters(self, how="random"):
        previous_layer = self.n
        i = 1
        for l in self.network_structure:
            if how == "he":
                mpr = np.sqrt(2/previous_layer)
            elif how == "random":
                mpr = 10
            else:
                mpr = 1
            #endif
            current_weights = np.random.randn(l, previous_layer)*mpr
            previous_layer = l
            self.weights[i] = current_weights
            self.bias[i] = np.zeros((l, 1))
            i += 1            
    #end
    
    
    
    def scale(self):
        '''also called min-max scaling or normalization; 
            resulting data is between [0, 1],
            data has smaller std. dev., it can suppress the effect of outliers'''
        self.X = (self.X - self.X_min) / (self.X_max - self.X_min)
        if len(self.X_cv) != 0:
            self.X_cv = (self.X_cv - self.X_min) / (self.X_max - self.X_min)
        if len(self.X_test) != 0:
            self.X_test = (self.X_test - self.X_min) / (self.X_max - self.X_min)
    #end scale
    
    def standardize(self):
        '''also called Z-score; 
            resulting data has mean 0 and std. dev. 1 - standard normal distribution, 
            it does not have bounding range, outliers are not affected'''   
        mean = np.sum(self.X, axis=1, keepdims=True) / self.m        
        variance = np.sum((self.X-mean)**2, axis=1, keepdims=True) / self.m
        self.X = (self.X - mean) / variance
        np.nan_to_num(self.X, False)
        if len(self.X_cv) != 0:
            self.X_cv = (self.X_cv - mean) / variance
            np.nan_to_num(self.X_cv, False)
        if len(self.X_test) != 0:
            self.X_test = (self.X_test - mean) / variance
            np.nan_to_num(self.X_test, False)
    #end standarize


    def sigmoid(self, z):
        """Sigmoid function - multi-label 
            prediciton: sigmoid >= 0.5 result 1; sigmoid < 0.5 result 0;
        """
        return 1 / (1 + np.exp(-z))
    #end
    
    def d_sigmoid(self, z):
        """ derivative of the sigmoid function """
        zPrim = self.sigmoid(z)
        return zPrim*(1-zPrim)
    #end
    
    def softmax(self, z):
        """softmax - multi-class, single-label
        """
        t = np.exp(z - np.max(z))
        #t = np.exp(z)
        return t / np.sum(t, axis=0)
    #end
    
    def d_softmax(self, z):
        softmax = self.softmax(z)
        return softmax*(1-softmax)
    #end
    
    def tanh(self, z):
        """Tanh activation function
            prediction: tanh >= 0 result 1; tanh < 0 result 0;
        """
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    #end
    
    def d_tanh(self, z):
        return 1 - np.power(self.tanh(z), 2)
    #end


    def forward_propagate(self, kind="train", weights=False, bias=False, data_in = [], dropout_in=True):
        ''' X shape
            Y shape '''
        if len(data_in) != 0:
            X = data_in
        elif kind == "train":
            X = self.X
        elif kind == "test" or kind == "cv":
            X = getattr(self, "X_"+kind)
        else:
            return 0
        #end if
        
        if weights == False:
            weights = self.weights
        #end if
        
        if bias == False:
            bias = self.bias
        #end if
        
        # layersA[0] will be x; layersZ[0] is empty
        self.cache_a = []
        self.cache_z = []
        self.cache_z.append(np.ndarray([]))
        self.cache_a.append(X)
        # Iterate throught layers; e.g. range(2) = [0, 1]; 
        for i in range(1, len(self.network_structure)+1):
            # cache_z.shape = (n-next + 1, 1)
            # b's shape is (i+1,1) and it's added to the new matrix before activation fn is applied
            # it is broadcasted to all results along m axis
            layer_z = np.dot(weights[i], self.cache_a[i-1]) + bias[i]
            # Save layerZ in cache for back prop
            self.cache_z.append(layer_z)
            if i == len(self.network_structure):
                # Last layer, applybie softmax/sigmoid
                if self.last_layer_af == "softmax":
                    layer_a = self.softmax(layer_z)
                else:
                    layer_a = self.sigmoid(layer_z)
                #end if
            else:
                # Non-last are tanh/sigmoid
                layer_a = self.tanh(layer_z)
            #end if
            np.nan_to_num(layer_a, False)
            if(dropout_in and self.dropout and i < len(self.network_structure)):
                # if hidden units and dropout then apply dropout
                dx = np.random.rand(layer_a.shape[0], layer_a.shape[1]) < self.dropout_keep_prob
                layer_a = np.multiply(layer_a, dx)
                layer_a /= self.dropout_keep_prob
            #end if dropout
            
            self.cache_a.append(layer_a)
        #end for
    #end forwardPropagate
    
    def back_propagate(self, kind="train", Y_in=[]):
        ''' labels should be nxm '''
        if kind != "train" and kind != "test" and kind != "cv":
            return 0
        #end if
        if len(Y_in) != 0:
            labels = Y_in
        elif kind == "train":
            labels = self.Y
        elif kind == "test" or kind == "cv":
            labels = getattr(self, "Y_"+kind)
        else:
            return 0
        m = labels.shape[1]
        #end if
        # e.g. range(2, 0) = [2, 1]
        n_depth = len(self.network_structure)
        self.cache_dw = {}
        self.cache_db = {}
        prev_dz = []
        for i in range(n_depth, 0, -1):
            if i == n_depth:
                # last layer sigmoid / softmax
                dz = (self.cache_a[i] - labels)
            else:
                # i = 1; s1 contain n1 neurons; (n1, m) = (n1, n2) x (n2, m) * (n1, m)
                dz = np.dot(self.weights[i+1].T, prev_dz)*self.d_tanh(self.cache_z[i])
            prev_dz = dz
            self.cache_dw[i] = np.dot(dz, self.cache_a[i-1].T) / m
            self.cache_db[i] = np.sum(dz, axis=1, keepdims=True) / m
            # no dropout regularization
            if(self.regularization and not self.dropout):
                reg = (self.reg_lambda * self.weights[i]) / (2 * m)
            else:
                reg = 0;
            self.weights[i] = self.weights[i] - self.alpha * (self.cache_dw[i] + reg)
            self.bias[i] = self.bias[i] - self.alpha * self.cache_db[i]
    #end backPropagate
    
    def learn(self, gradient_check = False):
        iterate = True
        cv_err_prev = -1
        cv_err = -1
        i = 0
        batch_i = 0
        if(self.mini_batch_size_n != -1):
# test
            mb_size = np.power(2, self.mini_batch_size_n)
            batches_count = int(self.m / mb_size) + 1
        else:
            mb_size = 0
            batches_count = 0
        #end if            
        while iterate:
            # Early stopping condition
            i += 1
            if(self.early_stopping):
# test
                cv_err_prev = cv_err
                cv_err = self.get_cv_error()
                if(self.max_iterations <= i or (i > 10 and cv_err_prev != -1 and cv_err_prev <= cv_err)):
                    iterate = False
                #end if
            elif(i >= self.iterations):
                iterate = False
            #end if
            
            if(self.mini_batch_size_n == -1):
                # Batch gd
                self.batch_gd(i)
            else:
                # Mini-batch gd
                batch_i = (i-1) * mb_size
                self.mini_batch_gd(batches_count, mb_size, batch_i)
            #end if
            
            # optional gradient check
            if gradient_check == True and (iterate == False or i == 1):
# test
                grad_diffs = self.performance_measure.check_gradients()
                print("Gradient chack after %1d iterations: %10.10f" % (i, grad_diffs))
            #end if
            
        #end while
        
        self.performance_measure.performance("cv")
        if(self.mini_batch_size_n == -1):
            # batch gd
            print("Last train error: ", self.cache_train_error[-1])
        else:
            # mini-batch gd
            print("Last train error: ",self.get_train_error())
        #end if
        print("Last cv error: ", self.cache_cv_error[-1])
        plt.clf()
        plt.plot(self.cache_iterations, self.cache_train_error)
        plt.plot(self.cache_iterations, self.cache_cv_error)
        plt.legend(["train error", "cv error"], loc="upper left")
        plt.title("Error vs iterations")
        plt.xlabel("iterations")
        plt.ylabel("error")
        plt.show()
        self.save_parameters()
    #end learn
    
    def batch_gd(self, i):
        self.forward_propagate()
        self.back_propagate()
    
        if i%100 == 0 or i == 1:
            training_error = self.get_train_error()
            print("Training error after %1d iterations: %10.10f" % (i, training_error))
            self.cache_train_error.append(training_error)
            self.cache_cv_error.append(self.get_cv_error())
            #self.cache_test_error.append(self.get_test_error())
            self.cache_iterations.append(i)
        #end if
    #end batch_gd
    
    def mini_batch_gd(self, batches_count, mb_size, batch_i):       
        for j in range(1, batches_count):
            batch_i += 1
            idx_start = mb_size * (j-1)
            idx_end = self.m["train"] if j == batches_count else mb_size * j
            data_mb = self.data_samples["train"][:, idx_start:idx_end]
            labels_mb = self.labels["train"][:, idx_start:idx_end]
            self.forward_propagate(data_in=data_mb)
            cache_a_temp = self.cache_a
            cache_z_temp = self.cache_z
            if batch_i%5 == 0 or batch_i == 1:
                # cost 
                train_error_mb = self.performance_measure.loss(kind="train", labels_in=labels_mb, m_in=data_mb.shape[1])
                self.cache_train_error.append(train_error_mb)
                self.cache_cv_error.append(self.get_cv_error())
                self.cache_iterations.append(batch_i)
            #end if
            self.cache_z = cache_z_temp
            self.cache_a = cache_a_temp
            self.back_propagate(labels_in=labels_mb)
        #end for
    #end mini_batch_gd
    
    def get_train_error(self):
        pred = self.predict("train")
        return self.performance_measure.loss("train", pred)
    
    def get_test_error(self):
        pred = self.predict("test")
        return self.performance_measure.loss("test", pred)
    #end
    
    def get_cv_error(self):
        pred = self.predict("cv")
        return self.performance_measure.loss("cv", pred)
    #end
    
    def predict(self, kind="train", custom_data=[], output="raw"):
        self.forward_propagate(kind, data_in=custom_data, dropout_in=False)
        if(output == "raw"):    
            return self.cache_a[-1]
        elif(output == "boolean"):
            return (self.cache_a[-1] >= 0.5)
    #end predict
    
    def round(self, num=0):
        if (num-int(num))>=0.5:
            return math.ceil(num)
        else:
            return math.floor(num)
        #else
    #end 
#end neural_network


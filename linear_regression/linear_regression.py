import numpy as np
import matplotlib.pyplot as plt
import math

class linear_regression:
    # data with bias
    X = None
    X_cv = None
    X_test = None
    # raw x
    X_raw = None
    # normalized x
    X_normal = None
    
    Y = None
    Y_cv = None
    Y_test = None
    # number of rows
    m = 0
    m_cv = 0
    m_test = 0
    # number of features without bias
    n = None
    is_norm = None
    x_mean = None
    std_dev = None
    cost ={ 'mae': np.array([]), 'mse': np.array([]), 'rmse': np.array([]), 'mape': np.array([]),'mpe': np.array([]), 'cv_rmse': np.array([]) }
    theta = None
    iterations = 1500
    alpha = 0.01
    normalize_columns = []
    
    def __init__(self, data, Y, normalize=False, normalize_columns=[], alpha=0.01, iterations=1500, cv_size=0.0, test_size=0.2):
        self.alpha = alpha
        self.iterations = iterations
        self.normalize_columns = normalize_columns
        
        to_shuffle = np.concatenate((data, Y), axis=1)
        np.random.shuffle(to_shuffle)
        
        self.X_raw = to_shuffle[:, :-1]
        self.Y = to_shuffle[:, -1].reshape(-1, 1)
        
        self.m = self.X_raw.shape[0]
        bias = np.ones((self.m, 1));
        self.n = self.X_raw.shape[1]
        self.theta = np.zeros((self.n+1, 1))
        self.is_norm = (normalize and normalize_columns)
        if(self.is_norm):
            self.features_normalize()
            self.X = np.concatenate((bias, self.X_normal), 1)
        else:
            self.x_mean = np.sum(self.X_raw, 0) / self.m
            self.std_dev = np.std(self.X_raw, 0)  
            self.X = np.concatenate((bias, self.X_raw), 1)
          
        # split into cv and test
        if(cv_size and cv_size+test_size < 0.6):
            self.m_cv = math.ceil(self.m * cv_size)
            self.X_cv = self.X[:self.m_cv, :]
            self.Y_cv = self.Y[:self.m_cv, :]
            
        if(test_size and cv_size+test_size < 0.6):
            self.m_test = math.ceil(self.m * test_size)
            self.X_test = self.X[self.m_cv:(self.m_cv + self.m_test), :]
            self.Y_test = self.Y[self.m_cv:(self.m_cv + self.m_test), :]
        
        self.m = self.m - self.m_cv - self.m_test
        self.X = self.X[(self.m_cv + self.m_test):, :]
        self.Y = self.Y[(self.m_cv + self.m_test):, :]
        
        print("Training samples: ", self.X.shape)
        if(self.m_cv): print("CV samples: ", self.X_cv.shape)
        if(self.m_test): print("Test samples: ", self.X_test.shape)
        
    #end
    
    def features_normalize(self, data_in=None): 
        """Normalizes features: (feature - mean) / standard deviation
            Add normalization by columns so that you can skip binary features.
        """
        if(self.x_mean is None):
            self.x_mean = np.sum(self.X_raw[:, self.normalize_columns], 0) / self.m
        if(self.std_dev is None):
            self.std_dev = np.std(self.X_raw[:, self.normalize_columns], 0)    
        raw_data = self.X_raw if(data_in is None) else data_in
        
        x_norm = ((raw_data[:, self.normalize_columns] - self.x_mean) / self.std_dev)
            
        x_norm_all = np.copy(raw_data)
        norm_col_i = 0
        for col_i in range(raw_data.shape[1]):
            if(col_i in self.normalize_columns):
                x_norm_all[:, col_i] = x_norm[:, norm_col_i]
                norm_col_i += 1

        if(self.X_normal is None):
            self.X_normal = x_norm_all

        return x_norm_all
    #end
    
    def plot(self, x_label, y_label):
        """Plot 2D input/output data
        """
        plt.scatter(self.X_raw, self.Y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    #end
    
    def error(self, kind='rmse', dataset='train', showDetails=False):
        if(dataset == 'cv'):
            X = self.X_cv
            Y = self.Y_cv
            
        elif(dataset == 'test'):
            X = self.X_test
            Y = self.Y_test
        else:
            X = self.X
            Y = self.Y
            
        if(showDetails):
            print(self.predict(X))
            print(Y)
            print((self.predict(X) - Y))
        if(kind == 'mae'):
            return self.mean_absolute_error(X, Y)
        elif(kind == 'mse'):
            return self.mean_square_error(X, Y)
        elif(kind == 'rmse'):
            return self.root_mean_square_error(X, Y)
        elif(kind == 'mape'):
            return self.mean_absolute_percentage_error(X, Y)
        elif(kind == 'mpe'):
            return self.mean_percentage_error(X, Y)
        #endif
    #end error
    
    def mean_absolute_error(self, X, Y):
        diff = (self.predict(X) - Y)
        return np.sum(diff) / (X.shape[0])
    #end
    
    def mean_square_error(self, X, Y):
        diffPow = np.power((self.predict(X) - Y), 2)
        return np.sum(diffPow) / (X.shape[0])
    #end
    
    def root_mean_square_error(self, X, Y):
        return np.sqrt(self.mean_square_error(X, Y))
    #end
    
    def mean_absolute_percentage_error(self, X, Y):
        pred = self.predict(X)
        pred[pred==0.] = np.finfo(np.float64).min
        return np.sum(np.abs((pred-Y)/pred)) / (X.shape[0])
    #end
    
    def mean_percentage_error(self, X, Y):
        pred = self.predict(X)
        pred[pred==0.] = np.finfo(np.float64).min
        return np.sum((pred-Y)/pred) / (X.shape[0])
    #end
    
    def plot_cost(self):
        """Plot cost agains number of iterations
        """
        plt.figure(1)
        plt.plot(self.cost['rmse'], label="rmse")
        plt.ylabel("cost rmse")
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
        
        plt.plot(self.cost['cv_rmse'], '-g', label="cv_rmse")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
        
# =============================================================================
#         plt.plot(self.cost['mae'], '-b', label="mae")
#         plt.plot(self.cost['rmse'], '-g', label="rmse")
#         plt.ylabel("cost")
#         plt.xlabel("iterations")
#         plt.legend()
#         plt.show()
# =============================================================================
        
# =============================================================================
#         plt.plot(self.cost['mape'], '--r', label="mape")
#         plt.plot(self.cost['mpe'], '--c', label="mpe")
#         plt.ylabel("cost")
#         plt.xlabel("iterations")
#         plt.legend()
#         plt.show()
# =============================================================================
    #end
    
    def gradient_descent(self):
        """Run gradient descent
        """
        for i in range(self.iterations):
            temp_cost_mse = self.error('mse')
            temp_cost_mae = self.error('mae')
            temp_cost_rmse = self.error('rmse')
            temp_cost_cv_rmse = self.error('rmse', dataset='cv' )
            if(i==0):
                print("Initial MSE: ", temp_cost_mse)
                print("Initial MAE: ", temp_cost_mae)
                print("Initial RMSE: ", temp_cost_rmse)
            self.cost['mse'] = np.append(self.cost['mse'], temp_cost_mse)
            self.cost['rmse'] = np.append(self.cost['rmse'], temp_cost_rmse)
            self.cost['mae'] = np.append(self.cost['mae'], temp_cost_mae)
            self.cost['mape'] = np.append(self.cost['mape'], self.error('mape'))
            self.cost['mpe'] = np.append(self.cost['mpe'], self.error('mpe'))
            self.cost['cv_rmse'] = np.append(self.cost['cv_rmse'], temp_cost_cv_rmse)
            temp_diff_out = self.X.dot(self.theta) - self.Y
            temp_diff_in = None
            
            for j in range(self.n+1):
                if j!=0:
                    # if we're not calculating thetas for bias term
                    temp_diff_in = temp_diff_out * self.X[:,[j]] 
                else:
                    temp_diff_in = temp_diff_out
                    
                self.theta[j] = self.theta[j] - ((self.alpha * np.sum(temp_diff_in)) / self.m)
            #end for    
                
        self.plot_cost()
        print("Final MSE: ",self.cost['mse'][-1])
        print("Final RMSE: ",self.cost['rmse'][-1])
        print("Final MAE: ",self.cost['mae'][-1])
        #print("Final theta: ", self.theta)
    #end
    
    def predict(self, data_in, data_with_bias=True, data_normalized=True):
        """ data_in: has to be normalized if the training data was
            with_bias: does the data have bias
        """
        will_normalize = self.is_norm and not data_normalized
        # Remove bias before normalization - if present
        if(data_with_bias & will_normalize):
            data_in = data_in[:, 1:]
            
        if(will_normalize):
            data_out = self.features_normalize(data_in)
            data_out = np.insert(data_out, 0, 1, axis=1)
        else:
            data_out = data_in
        #end if
        return data_out.dot(self.theta)
    #end    
    
    def print_data(self, limit=10):
        for i in range(0,limit):
            print("X: ",self.X_raw[i],";     y: ",self.Y[i])
    #end
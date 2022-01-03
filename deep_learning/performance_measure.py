# -*- coding: utf-8 -*-
"""
@author: al2357
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import float64, int
import math
import copy


class performance_measure:
  
    nn = None
    
    def __init__(self, nn):
        self.nn = nn
    #end __init__
    
    def check_gradients(self):
        epsilon = 0.0000001 # 10^(-7) as recommended by Andrew Ng
        theta_approx = np.array([])
        theta_calc = np.array([])

        # for each i(layer)
        for i in range(len(self.nn.weights), 0, -1):
            
            # calculate db and dW separately
            for j in range(0,2):
                
                if (j == 0):
                    # derivative approximation for weights_i
                    for x in range(0, self.nn.weights[i].shape[1]):
                        for y in range(0, self.nn.weights[i].shape[0]):                            
                            weights_copy_plus = copy.deepcopy(self.nn.weights)
                            bias_copy = copy.deepcopy(self.nn.bias)
                            weights_copy_minus = copy.deepcopy(self.nn.weights)
                            bias_copy = copy.deepcopy(self.nn.bias)
                            
                            
                            weights_copy_plus[i][y][x] = weights_copy_plus[i][y][x] + epsilon
                            weights_copy_minus[i][y][x] = weights_copy_minus[i][y][x] - epsilon
                            
                            # todo: to improve speed don't calculate on all training samples
                            self.nn.forward_propagate(weights=weights_copy_plus, bias=bias_copy, dropout_in=False)
                            predict_plus = self.nn.cache_a[-1]
                            plus_err = self.loss('train', predict_plus)
                                  
                            self.nn.forward_propagate(weights=weights_copy_minus, bias=bias_copy, dropout_in=False)
                            predict_minus = self.nn.cache_a[-1]
                            minus_err = self.loss('train', predict_minus)
          
                            theta_approx = np.append(theta_approx, ((plus_err - minus_err) / (2 * epsilon)))
                            theta_calc = np.append(theta_calc, self.nn.cache_dw[i][y][x])
                        # end for y
                        if x%50 == 0:
                            print("GC count; weights x: ", x)
                    # end for x
                    
                elif (j == 1):
                    for x in range(i, self.nn.bias[i].shape[0]):
                        # derivative approximation for bias_i
                        weights_copy = copy.deepcopy(self.nn.weights)
                        bias_copy_plus = copy.deepcopy(self.nn.bias)
                        weights_copy = copy.deepcopy(self.nn.weights)
                        bias_copy_minus = copy.deepcopy(self.nn.bias)
                        bias_copy_plus[i][x] = bias_copy_plus[i][x] + epsilon
                        bias_copy_minus[i][x] = bias_copy_minus[i][x] - epsilon
                        
                        self.nn.forward_propagate(weights=weights_copy, bias=bias_copy_plus, dropout_in=False)
                        predict_plus = self.nn.cache_a[-1]
                        plus_err = self.loss('train', predict_plus)
                              
                        self.nn.forward_propagate(weights=weights_copy, bias=bias_copy_minus, dropout_in=False)
                        predict_minus = self.nn.cache_a[-1]
                        minus_err = self.loss('train', predict_minus)
                        
                        theta_approx = np.append(theta_approx, ((plus_err - minus_err) / (2 * epsilon)))
                        theta_calc = np.append(theta_calc, self.nn.cache_db[i][x])
                    #end for x
                    print("GC count; bias x: ", x)
                #end if

             #end for range(0,2)
        #end for range(self.nn.weights.size, 0, -1)
        
        #euclidean_dist = np.linalg.norm(theta_approx - theta_calc)
        #between estimated gradients and backpropagation ones
        euclidean_dist = np.linalg.norm(theta_approx - theta_calc)
        norm = np.linalg.norm(theta_approx) + np.linalg.norm(theta_calc)
        grad_diffs = (euclidean_dist / norm)
        
        return grad_diffs
    #end check_gradient  
    
    def loss(self, kind='train', pred_in = [], labels_in = [], m_in = None):
        if len(labels_in) != 0:
            labels = labels_in
        elif kind == 'train':
            labels = self.nn.Y
        elif kind == 'test' or kind == 'cv':
            labels = getattr(self.nn, "Y_"+kind)
        else:
            return 0
        #end if

        if len(pred_in) != 0:
            pred = pred_in
        elif kind == 'train':
            pred = self.nn.cache_a[-1]
        else:
            return
        #end if
        
        if m_in != None:
            m_local = m_in
        elif kind == 'train':
            m_local = self.nn.m
        elif kind == 'test' or kind == 'cv': 
            m_local = getattr(self.nn, "m_"+kind)
        else:
            return
        #end if
        
        if (self.nn.regularization):
            weights_sum = 0
            for k in self.nn.weights:
                # L2
                weights_sum += np.sum(np.square(self.nn.weights[k]))
                # L1
                #weights_sum += np.sum(self.nn.weights[k])
            reg = (self.nn.reg_lambda * weights_sum) / (2 * m_local)
        else:
            reg = 0
        #end if
        
        if self.nn.last_layer_af == "softmax":
            return self.categorical_crossentropy(kind, labels, pred, reg, m_local)
        else:
            return self.binary_crossentropy(kind, labels, pred, reg, m_local)
        #end if
    #end loss

    def binary_crossentropy(self, kind, labels, pred, reg, m_in):
        '''sigmoid''' 
        product_true = labels*pred
        product_false = (1-labels)*pred
        product_true = product_true[product_true != 0]
        product_false = product_false[product_false != 0]
        loss = -(np.sum(np.log(product_true)) + np.sum(np.log(1-product_false)) / m_in) + reg
        return loss
    #end binary_crossentropy

    def categorical_crossentropy(self, kind, labels, pred, reg, m_in):
        '''softmax - multi-class single-label'''
        product = labels*pred
        loss = -(np.sum(np.log(product[product!=0])) / m_in) + reg
        return loss
    #end categorical_crossentropy
    
    def performance(self, kind, show_micro = False):
        ''' precision = tp / (tp + fp), 
            recall = tp / (tp + fn), 
            accuracy = (tp + tn) / (tp + fp + tn + fn), 
            fscore = 2*precision*recall / (precision+recall)
        
            "Macro-averaging treats all classes equally while micro-averaging favors bigger classes." - 
            "A systematic analysis of performance measures for classification tasks." M.Sokolova, G.Lapalme'''
        # Predict so that last layer of nn.cache_a is of a correct kind
        self.nn.predict(kind) 
        # Change probabilities to oh array - m x n
        pred_max_idxs = np.argmax(self.nn.cache_a[-1], axis=0)
        pred_oh = np.zeros(self.nn.cache_a[-1].shape)
        pred_oh[pred_max_idxs, np.arange(len(pred_max_idxs))] = 1
        if kind == 'train':
            labels = self.nn.Y
        elif kind == 'test' or kind == 'cv': 
            labels = getattr(self.nn, "Y_"+kind)
        
        if(show_micro == True):
            # Micro averaging - use confusion matrix of all classes to calculate PMs.
            tp_micro = np.sum((pred_oh[:, :] == 1) & (labels[:, :] == 1))
            fp_micro = np.sum((pred_oh[:, :] == 1) & (labels[:, :] == 0))
            tn_micro = np.sum((pred_oh[:, :] == 0) & (labels[:, :] == 0))
            fn_micro = np.sum((pred_oh[:, :] == 0) & (labels[:, :] == 1))
            
            precision_micro = tp_micro / (tp_micro + fp_micro)
            recall_micro = tp_micro / (tp_micro + fn_micro)
            accuracy_micro = (tp_micro + tn_micro) / (tp_micro + fp_micro + tn_micro + fn_micro)
            f1_micro = 2*precision_micro*recall_micro / (precision_micro + recall_micro)
            
            print("Precision micro: ", precision_micro)
            print("Recall micro: ", recall_micro)
            print("Accuracy micro: ", accuracy_micro)
            print("F1-score micro: ", f1_micro)
        #end if
        
        # Macro averaging - calculate PMs for each class and then get their average.
        precision_sum = 0
        recall_sum = 0
        acc_sum = 0
        for i in range(labels.shape[0]):
            tp = np.sum((pred_oh[i, :] == 1) & (labels[i, :] == 1))
            fp = np.sum((pred_oh[i, :] == 1) & (labels[i, :] == 0))
            tn = np.sum((pred_oh[i, :] == 0) & (labels[i, :] == 0))
            fn = np.sum((pred_oh[i, :] == 0) & (labels[i, :] == 1))
            precision_sum += tp / (tp+fp)
            recall_sum += tp / (tp+fn)
            acc_sum += (tp + tn) / (tp + fp + tn + fn)
        #end for
        
        precision_macro = precision_sum / labels.shape[0]
        recall_macro = recall_sum / labels.shape[0]
        acc_macro = acc_sum / labels.shape[0]

        print("Precision macro: ", precision_macro)
        print("Recall macro: ", recall_macro)
        print("Accuracy macro: ", acc_macro)        
        
        return [precision_macro, recall_macro, acc_macro]
    #end fscore
        
#end performance_measure
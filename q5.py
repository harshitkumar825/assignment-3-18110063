import pandas as pd
from autograd import grad
from autograd import grad, numpy as auto
from math import e

class MLP:
    def __init__(self, n_hidden_layers, a_hidden_layers, fit_type = "classification", n_features = 0, n_classes = 10):
        
        self.N = len(n_hidden_layers)
        if(self.N == 0):
            n_hidden_layers = [n_features]
        self.n_hidden_layers = n_hidden_layers
        self.a_hidden_layers = a_hidden_layers
        self.fit_type = fit_type
        self.n_classes = n_classes

        inp_size = n_features
        sw=[]
        sb=[]

        for i in range(self.N):
            sw.append( auto.array([[0.0]*inp_size]*self.n_hidden_layers[i]).T  )
            sb.append( auto.array([0.0]*self.n_hidden_layers[i]).T  )
            inp_size = self.n_hidden_layers[i]
        if(fit_type == "classification"):
            sw.append( auto.array([[0.0]*self.n_hidden_layers[-1]]*self.n_classes).T  )
            sb.append( auto.array([0.0]*self.n_classes).T  )
        else:
            sw.append( auto.array([[0.0]*self.n_hidden_layers[-1]]*1).T  )
            sb.append( auto.array([0.0]).T  )
        self.WEIGHTS=sw
        self.BIASES=sb
        return

    def softmax(self,X):
        P = auto.exp(X)
        return P/auto.sum(P,axis=1).reshape(-1,1)

    def forwardprop(self, X, WEIGHTS, BIASES):
        self.X = X
        inp_next = X
        for i in range(self.N):
            output =  auto.dot(auto.array(inp_next),WEIGHTS[i]) + auto.array([BIASES[i]]*inp_next.shape[0])
            activation = self.a_hidden_layers[i]
            if(activation == "sigmoid"):
                output = (1.0)/(1+e**(-auto.array(output)))
            elif(activation == "relu"):
                output = auto.maximum(output, 0.0)
            else:
                pass
            inp_next = output

        output =  auto.dot(auto.array(inp_next),WEIGHTS[-1]) + auto.array([BIASES[-1]]*inp_next.shape[0])
        
        if(self.fit_type == "classification"):
            output = self.softmax(output)
        else:
            output = auto.maximum(output, 0.0)

        return output
    
    def class_err_func(self, weights, biases, y):
        y_hat = self.forwardprop(self.X, weights, biases)
        class_err = 0
        for k in range(self.n_classes):
            class_err -= auto.dot((y == k).astype(float),auto.log(y_hat[:,k]))
        return class_err

    def mse_func(self, weights, biases, y):
        y_hat = self.forwardprop(self.X, weights, biases)
        MSE = auto.sum(auto.square(auto.subtract(y_hat, y.reshape(-1,1))))/len(y)
        return MSE

    def backprop(self, lr, y):
        if(self.fit_type =="classification"):
            dJw = grad(self.class_err_func,0)(self.WEIGHTS,self.BIASES, y)
            dJb = grad(self.class_err_func,1)(self.WEIGHTS,self.BIASES, y)
        else:
            dJw = grad(self.mse_func,0)(self.WEIGHTS,self.BIASES, y)
            dJb = grad(self.mse_func,1)(self.WEIGHTS,self.BIASES, y)
        
        for i in range(self.N + 1):
            self.WEIGHTS[i] -= lr*dJw[i]/len(self.X)
            self.BIASES[i] -= lr*dJb[i]/len(self.X)
        return

    def predict(self, X):
        y_hat = self.forwardprop(X, self.WEIGHTS, self.BIASES)
        if(self.fit_type =="classification"):
            return auto.argmax(y_hat,axis=1)
        return y_hat


    

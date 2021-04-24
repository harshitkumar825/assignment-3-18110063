import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad, numpy as auto
from metrics import *
from sklearn.datasets import load_breast_cancer
from math import e

np.random.seed(42)

class UnregularizedLogisticRegression():
    def __init__(self,fit_intercept=True):
        self.coef_ = None
        self.fit_intercept=fit_intercept
        
    def fit(self, X, y, batch_size=-1, n_iter=100, lr=0.01, lr_type='constant'):
        samples=len(X)
        if batch_size==-1 or batch_size>samples:
            batch_size=samples
        if self.fit_intercept:
            X=pd.concat([pd.Series([1]*samples),X],axis=1,ignore_index=True)
        self.iterations=n_iter
        alpha=lr
        theta=np.array([0.0]*len(X.columns)).T
        for i in range(self.iterations):
            if lr_type=='inverse':
                alpha=lr/(i+1)
            start=(i*batch_size)%samples
            end=((i+1)*batch_size-1)%samples+1
            if end<start:
                X_batch=pd.concat([X.iloc[start:],X.iloc[:end]],axis=0,ignore_index=True)
                y_batch=pd.concat([y.iloc[start:],y.iloc[:end]],axis=0,ignore_index=True)
            else:
                X_batch=X.iloc[start:end]
                y_batch=y.iloc[start:end]
            new_theta=np.array([0.0]*len(X.columns)).T
            for j in range(len(X.columns)):
                djj=0
                for k in range(batch_size):
                    theta_dot_xi=0
                    for l in range(len(X.columns)):
                        theta_dot_xi+=theta[l]*(X_batch.iloc[k,l])
                    djj+=(1/(1+np.exp(-theta_dot_xi))-y_batch.iloc[k])*X_batch.iloc[k,j]
                new_theta[j]=theta[j]-alpha*djj/batch_size
            theta=new_theta
        self.coef_=theta

    def sigmoid(self,y_cap):
        return 1.0/(1+e**(-auto.array(y_cap)))
    
    def automle(self,theta):
        sigm=self.sigmoid(auto.dot(np.array(self.xbatch),theta))
        ans=-(auto.dot(self.ybatch.T,auto.log(sigm))+auto.dot((auto.ones(self.ybatch.shape)-self.ybatch).T,auto.log(auto.ones(self.ybatch.shape)-sigm)))
        return ans

    def fit_autograd(self, X, y, batch_size=-1, n_iter=100, lr=0.01, lr_type='constant'):
        samples=len(X)
        if batch_size==-1 or batch_size>samples:
            batch_size=samples
        if self.fit_intercept:
            X=pd.concat([pd.Series([1]*samples),X],axis=1,ignore_index=True)
        self.iterations=n_iter
        alpha=lr
        theta=np.array([0.0]*len(X.columns)).T
        for i in range(self.iterations):
            if lr_type=='inverse':
                alpha=lr/(i+1)
            start=(i*batch_size)%samples
            end=((i+1)*batch_size-1)%samples+1
            if end<start:
                X_batch=pd.concat([X.iloc[start:],X.iloc[:end]],axis=0,ignore_index=True)
                y_batch=pd.concat([y.iloc[start:],y.iloc[:end]],axis=0,ignore_index=True)
            else:
                X_batch=X.iloc[start:end]
                y_batch=y.iloc[start:end]
            self.xbatch=X_batch
            self.ybatch=y_batch
            grad_dmle=grad(self.automle)
            dmle=grad_dmle(theta)
            theta=theta-(alpha/batch_size)*dmle
        self.coef_=theta


    def predict(self,X):
        if self.fit_intercept:
            X=pd.concat([pd.Series([1]*len(X)),X],axis=1,ignore_index=True)
        y_cap=pd.Series(X.dot(self.coef_))
        y_cap[y_cap<0]=0
        y_cap[y_cap>0]=1
        return y_cap

    def plot_decision_boundary(self, X, y,title):
        c,w1,w2=list(self.coef_)
        slope=-w1/w2
        c/=-w2
        X_min,X_max,y_min,y_max=-2,2,-2,2
        Xs=np.array([X_min, X_max])
        ys=slope*Xs + c
        plt.figure()
        plt.plot(Xs, ys, 'k', lw=1, ls='-')
        plt.fill_between(Xs, ys, y_min, color='orange', alpha=0.2)
        plt.fill_between(Xs, ys, y_max, color='blue', alpha=0.2)
        plt.scatter(X[y==0][0],X[y==0][1],s=8,cmap='Paired')
        plt.scatter(X[y==1][0],X[y==1][1],s=8,cmap='Paired')
        plt.xlim(X_min, X_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(title)
        plt.show()

############################################################################################
        
N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
X = (X-X.min())/(X.max()-X.min())
y = pd.Series(np.random.randint(2,size=N))
fit_intercept = True

############################################################################################

print("\n-->Unregularised Logistic Regression using defined gradient descent")
LR=UnregularizedLogisticRegression(fit_intercept=fit_intercept)
LR.fit(X, y, n_iter=200,batch_size=5,lr=2.5)
y_hat=LR.predict(X)
print("Obtained theta values : ", LR.coef_)
print('Accuracy : ', accuracy(y_hat, y))

LR.plot_decision_boundary(X,y,'Unregularized Logistic Regression with defined Gradient Descent')


print("\n-->Unregularised Logistic Regression using Autograd")
LR = UnregularizedLogisticRegression(fit_intercept=fit_intercept)
LR.fit_autograd(X,y,n_iter=200,batch_size=5,lr=2.5)
y_hat=LR.predict(X)
print("Obtained theta values : ", LR.coef_)
print('Autograd Accuracy : ', accuracy(y_hat, y))
LR.plot_decision_boundary(X,y,'Unregularized Logistic Regression with Gradient Descent using Autograd')
############################################################################################
print("\n-->3-Fold Unregularised Logistic Regression on Breast Cancer Dataset")
X,y=load_breast_cancer(return_X_y=True,as_frame=True)
X=(X-X.min())/(X.max()-X.min())
data=pd.concat([X, y.rename("y")],axis=1,ignore_index=True)
data=data.sample(frac=1).reset_index(drop=True)
FOLDS=3
size=len(data)//FOLDS
Xfolds=[data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds=[data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
avg_accuracy=0
for i in range(FOLDS):
    print("Test_fold =",i+1)
    Xdash,ydash=Xfolds.copy(),yfolds.copy()
    X_test,y_test=Xdash[i],ydash[i]
    X_test.index=[q for q in range(len(X_test))]
    X_test.columns=[q for q in range(len(X_test.iloc[0]))]
    y_test.index=[q for q in range(len(y_test))]
    Xdash.pop(i)
    ydash.pop(i)
    X_train,y_train=pd.concat(Xdash),pd.concat(ydash)
    X_train,y_train=pd.DataFrame(X_train),pd.Series(y_train)
    X_train.index=[q for q in range(len(X_train))]
    X_train.columns=[q for q in range(len(X_train.iloc[0]))]
    y_train.index=[q for q in range(len(y_train))]
    LR=UnregularizedLogisticRegression(fit_intercept=True)
    LR.fit_autograd(X_train,y_train,n_iter=1000,batch_size=25,lr=3,lr_type="inverse")
    y_hat=LR.predict(X_test)
    test_accuracy=accuracy(y_hat,y_test.reset_index(drop=True))
    print("\tTest_Accuracy :",test_accuracy)
    avg_accuracy+=test_accuracy
avg_accuracy=avg_accuracy/FOLDS
print("Average Accuracy :",avg_accuracy)
    

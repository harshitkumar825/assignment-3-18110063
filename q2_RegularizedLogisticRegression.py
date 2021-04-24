import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad, numpy as auto
from metrics import *
from sklearn.datasets import load_breast_cancer
from math import e

class RegularizedLogisticRegression():
    def __init__(self,fit_intercept=True):
        self.coef_ = None
        self.fit_intercept=fit_intercept

    def sigmoid(self,y_cap):
        return 1.0/(1+e**(-auto.array(y_cap)))
    
    def l1_auto(self,theta):
        sigm=self.sigmoid(auto.dot(np.array(self.xbatch),theta))
        ans=-(auto.dot(self.ybatch.T,auto.log(sigm))+auto.dot((auto.ones(self.ybatch.shape)-self.ybatch).T,auto.log(auto.ones(self.ybatch.shape)-sigm)))+self.lmbda*auto.sum(auto.abs(theta))
        return ans
        
    def l1_regularised_autograd(self, X, y, batch_size=-1, n_iter=100, lr=0.01, lr_type='constant',lmbda=1):
        self.lmbda=lmbda
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
            grad_dmle=grad(self.l1_auto)
            dmle=grad_dmle(theta)
            theta=theta-(alpha/batch_size)*dmle
        self.coef_=theta


    def l2_auto(self,theta):
        sigm=self.sigmoid(auto.dot(np.array(self.xbatch),theta))
        ans=-(auto.dot(self.ybatch.T,auto.log(sigm))+auto.dot((auto.ones(self.ybatch.shape)-self.ybatch).T,auto.log(auto.ones(self.ybatch.shape)-sigm)))+self.lmbda*auto.dot(theta,theta)
        return ans
        
    def l2_regularised_autograd(self, X, y, batch_size=-1, n_iter=100, lr=0.01, lr_type='constant',lmbda=1):
        self.lmbda=lmbda
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
            grad_dmle=grad(self.l2_auto)
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

    
    def plot_decision_boundary(self, X, y):
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
        plt.show()

print("\n-->Nested cross-validation for finding Optimal lambda values")
X,y=load_breast_cancer(return_X_y=True,as_frame=True)
X=(X-X.min())/(X.max()-X.min())
data=pd.concat([X, y.rename("y")],axis=1,ignore_index=True)
data=data.sample(frac=1).reset_index(drop=True)
FOLDS=3
size=len(data)//FOLDS
Xfolds=[data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds=[data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
cross_val_folds=4
lmbdas=list(np.arange(0.5,5,0.5))
for reg_type in ["l1","l2"]:
    Optimals=[]
    print(f"\n-->Regularization Type {reg_type}")
    for i in range(FOLDS):
        print("Test fold =",i+1)
        Xdash, ydash = Xfolds.copy(), yfolds.copy()
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
        size=len(X_train)//cross_val_folds
        X_train_folds=[X_train.iloc[j*size:(j+1)*size] for j in range(cross_val_folds)]
        y_train_folds=[y_train.iloc[j*size:(j+1)*size] for j in range(cross_val_folds)]
        val_accuracies=[]

        for lmbda in lmbdas:
            print("\tlambda = ",lmbda)
            avg_validation_accuracy=0
            for k in range(cross_val_folds):
                X_traindash, y_traindash = X_train_folds.copy(), y_train_folds.copy()
                X_valid,y_valid=X_train_folds[k],y_train_folds[k]
                X_valid=X_valid.reset_index(drop=True)
                y_valid=y_valid.reset_index(drop=True)
                X_traindash.pop(k)
                y_traindash.pop(k)
                train_X,train_y= pd.concat(X_traindash),pd.concat(y_traindash)
                train_X=train_X.reset_index(drop=True)
                train_y=train_y.reset_index(drop=True)
                LR=RegularizedLogisticRegression()
                if reg_type=='l1':
                    LR.l1_regularised_autograd(train_X, train_y, n_iter=200,lr=0.5, lmbda=lmbda,lr_type="inverse")
                else:
                    LR.l2_regularised_autograd(train_X, train_y, n_iter=200,lr=0.5, lmbda=lmbda,lr_type="inverse")
                y_hat=LR.predict(X_valid.reset_index(drop=True))
                valid_accuracy = accuracy(y_hat,y_valid.reset_index(drop=True))
                print("\t\tValidation Fold =",k + 1,", Accuracy : ",valid_accuracy)
                avg_validation_accuracy+=valid_accuracy
            avg_validation_accuracy=avg_validation_accuracy/cross_val_folds
            print("\t\t\tAvg_val_accuracy :",avg_validation_accuracy)
            val_accuracies.append(avg_validation_accuracy)
        opt_lmbda=0
        opt_acc=0
        for i in range(len(val_accuracies)):
            if(val_accuracies[i]>=opt_acc):
                opt_lmbda = lmbdas[i]
                opt_acc = val_accuracies[i]

        print("\tOptimal lambda :",opt_lmbda,", Optimal_Accuracy :", opt_acc)
        Optimals.append(opt_lmbda)
            
    print("The optimal lamdas for each folds are ", Optimals)

print("\n-->Feature Selection using L1 Regularisation")
THETAS = []
lambdas = list(np.arange(0.0,25,0.1))

for lmbda in lambdas:
    LR=RegularizedLogisticRegression(fit_intercept=False)
    LR.l1_regularised_autograd(X, y, n_iter=100, lr = 0.5, batch_size=len(X), lmbda = lmbda)
    THETAS.append(np.array(LR.coef_))
THETAS = np.array(THETAS)

plt.figure()
for col in range(10):
    plt.plot(lambdas,THETAS[:,col],label=X.columns[col])

plt.xlabel('Lambda') 
plt.ylabel('Theta') 
plt.title('Feature Selection using L1 Regularisation') 
plt.legend()
plt.show()

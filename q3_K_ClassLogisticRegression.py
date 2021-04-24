import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad, numpy as auto
from sklearn.preprocessing import MinMaxScaler
from metrics import *
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
np.random.seed(42)

class K_ClassLogisticRegression():
    def __init__(self,fit_intercept=True):
        self.coef_ = None
        self.fit_intercept=fit_intercept

    def softmax(self, X, k, theta):
        p=auto.exp(auto.dot(X,theta))
        ans=p[:,k]/auto.sum(p,axis=1)
        return ans

    def fit_multi(self, X, y, batch_size=-1, n_iter=100, lr=0.01, lr_type='constant'):
        self.X=X
        samples=len(X)
        if batch_size==-1 or batch_size>samples:
            batch_size=samples
        if self.fit_intercept:
            X=pd.concat([pd.Series([1]*samples),X],axis=1,ignore_index=True)
        self.iterations=n_iter
        alpha=lr
        classes=sorted(list(y.unique()))
        self.classes = classes
        theta=np.array([[0.0]*len(X.columns)]*len(classes)).T
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
            new_theta=np.array([0]*len(X.columns)).T
            for k in classes:
                loss=-((y_batch==k).astype(float)-self.softmax(X_batch,k,theta))
                theta[:,k]=theta[:,k]-(alpha/batch_size)*X_batch.T.dot(loss)
        self.coef_=theta

    def autocross(self,theta):
        p=auto.exp(auto.dot(np.array(self.X),theta))
        p/=auto.sum(p,axis=1).reshape(-1,1)
        c=0
        for k in self.classes:
            c-=auto.dot((self.y==k).astype(float),auto.log(p[:,k]))
        return c

    def fit_multi_autograd(self, X, y, batch_size=-1, n_iter=100, lr=0.01, lr_type='constant'):
        samples=len(X)
        if batch_size==-1 or batch_size>samples:
            batch_size=samples
        if self.fit_intercept:
            X=pd.concat([pd.Series([1]*samples),X],axis=1,ignore_index=True)
        self.iterations=n_iter
        self.X=X
        self.y=y
        alpha=lr
        classes=sorted(list(y.unique()))
        self.classes=classes
        theta=np.array([[0.0]*len(X.columns)]*len(classes)).T
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
            grad_autocross=grad(self.autocross)
            ac=grad_autocross(theta)
            theta=theta-(alpha/batch_size)*ac
        self.coef_=theta

    def predict_multi(self,X):
        if self.fit_intercept:
            X=pd.concat([pd.Series([1]*len(X)),X],axis=1,ignore_index=True)
        y_cap=np.zeros_like(X.dot(self.coef_))
        for k in self.classes:
            y_cap[:,k]=self.softmax(X,k,self.coef_)
        return np.argmax(y_cap,axis=1)


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

###################################################

X, y = load_digits(return_X_y=True,as_frame=True)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
split = int(0.7*len(data)) # TRAIN-TEST SPLIT
X_train, y_train = data.iloc[:split].iloc[:,:-1], data.iloc[:split].iloc[:,-1]
X_train.index=[q for q in range(len(X_train))]
X_train.columns=[q for q in range(len(X_train.iloc[0]))]
y_train.index=[q for q in range(len(y_train))]
X_test, y_test = data.iloc[split:].iloc[:,:-1], data.iloc[split:].iloc[:,-1]
X_test.index=[q for q in range(len(X_test))]
X_test.columns=[q for q in range(len(X_test.iloc[0]))]
y_test.index=[q for q in range(len(y_test))]

print("\n-->Multi-class Logistic Regression using self-update rules")

LR = K_ClassLogisticRegression()
LR.fit_multi(X_train, y_train, n_iter=100,batch_size = len(X_train),lr = 3)
y_hat = LR.predict_multi(X_test)
print('Accuracy: ', accuracy(y_hat, y_test))

print("\n-->Multi-class Logistic Regression using Autograd")

LR = K_ClassLogisticRegression()
LR.fit_multi_autograd(X_train, y_train,n_iter=100,lr=3)
y_hat = LR.predict_multi(X_test)
print('Accuracy :', accuracy(y_hat, y_test))

###################################################

print("\n-->4-Folds Multi-class Logistic Regression over DIGITS")

X, y = load_digits(return_X_y=True,as_frame=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
FOLDS = 4
size = len(data)//FOLDS
Xfolds = [data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds = [data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
avg_accuracy = 0
accs = []
for i in range(FOLDS):
    print("Test_fold =",i+1)
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
    LR = K_ClassLogisticRegression()
    LR.fit_multi(X_train, y_train, n_iter=500,batch_size = 20,lr = 3, lr_type="inverse")
    y_hat = LR.predict_multi(X_test)
    test_accuracy = accuracy(y_hat,y_test.reset_index(drop=True))
    accs.append([test_accuracy,y_hat, y_test])
    print("\tTest_Accuracy :",test_accuracy)
    avg_accuracy += test_accuracy

avg_accuracy = avg_accuracy/FOLDS
print("AVERAGE ACCURACY =",avg_accuracy)

accs = sorted(accs, key=lambda x: x[0], reverse=True)
best_y_hat = accs[0][1]
best_y_test = accs[0][2]

print("\n-->Best Confusion Matrix --------")
cm=confusion_matrix(np.array(best_y_test), np.array(best_y_hat))
print(cm)
df_cm=pd.DataFrame(cm, index = [i for i in "0123456789"],columns = [i for i in "0123456789"])
plt.figure()
sns.heatmap(df_cm, annot=True)

plt.show()
print("\n-->Principal Component Analysis (PCA) for DIGITS")

digits = load_digits()
plt.figure()
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c = digits.target, cmap="tab10")
plt.colorbar()
plt.show()

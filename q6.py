import pandas as pd
from autograd import grad,  numpy as auto
from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import MinMaxScaler
from metrics import *
from math import e
from q5 import MLP
import matplotlib.pyplot as plt
auto.random.seed(42)

print("\n-->3-Fold NN CLassification on DIGITS Dataset")

X,y=load_digits(return_X_y=True,as_frame=True)
scaler=MinMaxScaler()
X=pd.DataFrame(scaler.fit_transform(X))
y=pd.Series(y)
data=pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data=data.sample(frac=1).reset_index(drop=True)

FOLDS=3
size=len(data)//FOLDS
Xfolds=[data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds=[data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
avg_accuracy=0
plt.figure(1)
plt.title('Classification on NN using 3-fold cv')
for i in range(FOLDS):
    print("Test_fold =",i+1)
    Xdash, ydash=Xfolds.copy(), yfolds.copy()
    X_test, y_test=Xdash[i], ydash[i]
    X_test.index=[q for q in range(len(X_test))]
    X_test.columns=[q for q in range(len(X_test.iloc[0]))]
    y_test.index=[q for q in range(len(y_test))]
    Xdash.pop(i)
    ydash.pop(i)
    X_train,y_train=pd.concat(Xdash), pd.concat(ydash)
    X_train.index=[q for q in range(len(X_train))]
    X_train.columns=[q for q in range(len(X_train.iloc[0]))]
    y_train.index=[q for q in range(len(y_train))]
    NN=MLP([25],['sigmoid'],"classification",X_train.shape[1],len(list(y_train.unique())))
    epochs=list(range(100))
    lr=2
    losses=[]
    for epoch in epochs:
        output=NN.forwardprop(X_train, NN.WEIGHTS, NN.BIASES)
        epoch_loss=NN.class_err_func(NN.WEIGHTS, NN.BIASES, y_train)
        losses.append(epoch_loss)
        NN.backprop(lr, auto.array(y_train))
    plt.plot(epochs,losses,label='Fold'+str(i+1))
    y_hat=NN.predict(X_test)
    test_accuracy=accuracy(y_hat, y_test)
    print('Accuracy', test_accuracy)
    avg_accuracy += test_accuracy
avg_accuracy=avg_accuracy/FOLDS
print("AVERAGE ACCURACY =",avg_accuracy)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

print("\n-->3-Fold NN Regression on BOSTON Dataset")


X,y=load_boston(return_X_y=True)
scaler=MinMaxScaler()
X=pd.DataFrame(scaler.fit_transform(X))
y=pd.Series(y)
data=pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data=data.sample(frac=1).reset_index(drop=True)
FOLDS=3
size=len(data)//FOLDS
Xfolds=[data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds=[data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
avg_rmse=0
plt.figure(2)
plt.title('Regression on NN using 3-fold cv')
for i in range(FOLDS):
    print("Test_fold =",i+1)
    Xdash, ydash=Xfolds.copy(), yfolds.copy()
    X_test, y_test=Xdash[i], ydash[i]
    X_test.index=[q for q in range(len(X_test))]
    X_test.columns=[q for q in range(len(X_test.iloc[0]))]
    y_test.index=[q for q in range(len(y_test))]
    Xdash.pop(i)
    ydash.pop(i)
    X_train,y_train=pd.concat(Xdash), pd.concat(ydash)
    X_train.index=[q for q in range(len(X_train))]
    X_train.columns=[q for q in range(len(X_train.iloc[0]))]
    y_test.index=[q for q in range(len(y_test))]
    NN=MLP([15, 8],['sigmoid', 'relu'],'regression',X_train.shape[1],len(list(y_train.unique())))
    epochs=list(range(70))
    lr=1
    losses=[]
    for epoch in epochs:
        output=NN.forwardprop(X_train, NN.WEIGHTS, NN.BIASES)
        epoch_loss=NN.mse_func(NN.WEIGHTS, NN.BIASES, auto.array(y_train))
        losses.append(epoch_loss)
        NN.backprop(lr, auto.array(y_train))
    plt.plot(epochs,losses,label='Fold'+str(i+1))
    y_hat=NN.predict(X_test)
    test_rmse=rmse(y_hat, y_test)
    print('RMSE :', test_rmse)
    avg_rmse += test_rmse
avg_rmse=avg_rmse/FOLDS
print("AVERAGE ACCURACY =",avg_rmse)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

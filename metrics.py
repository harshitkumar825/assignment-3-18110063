def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size),'Size Mismatch'
    ans=0
    # TODO: Write here
    for i in range(y.size):
        if str(float(y_hat[i]))==str(float(y[i])):
            ans+=1
    return ans/y.size

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size),'Size Mismatch'
    ans=0
    den=0
    # TODO: Write here
    for i in range(y.size):
        if str(y_hat[i])==str(cls):
            den+=1
            if str(y[i])==str(cls):
                ans+=1
    if den==0:
        return 'N/A'
    return ans/den

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size),'Size Mismatch'
    ans=0
    den=0
    # TODO: Write here
    for i in range(y.size):
        if str(y[i])==str(cls):
            den+=1
            if str(y_hat[i])==str(cls):
                ans+=1
    return ans/den

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    assert(y_hat.size == y.size),'Size Mismatch'
    ans=0
    for i in range(y.size):
        ans+=(float(y_hat[i])-float(y[i]))**2
    return (ans/y.size)**0.5

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size),'Size Mismatch'
    ans=0
    for i in range(y.size):
        if float(y_hat[i])>float(y[i]):
            ans+=float(y_hat[i])-float(y[i])
        else:
            ans+=float(y[i])-float(y_hat[i])
    return ans/y.size

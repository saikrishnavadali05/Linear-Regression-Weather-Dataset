# Linear Regression from Scratch without Sklearn

#importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The method "split_data" splits the given dataset into trainset and testset
# This is similar to the method "train_test_split" from "sklearn.model_selection"
def split_data(X,y,test_size=0.2,random_state=0):
    np.random.seed(random_state)                  #set the seed for reproducible results
    indices = np.random.permutation(len(X))       #shuffling the indices
    data_test_size = int(X.shape[0] * test_size)  #Get the test size

    #Separating the Independent and Dependent features into the Train and Test Set
    train_indices = indices[data_test_size:]
    test_indices = indices[:data_test_size]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test

class linearRegression():

  def __init__(self):
    #No instance Variables required
    pass

  def forward(self,X,y,W):
    """
    Parameters:
    X (array) : Independent Features
    y (array) : Dependent Features/ Target Variable
    W (array) : Weights 

    Returns:
    loss (float) : Calculated Sqaured Error Loss for y and y_pred
    y_pred (array) : Predicted Target Variable
    """

    y_pred = sum(W * X)
    loss = ((y_pred-y)**2)/2    #Loss = Squared Error, we introduce 1/2 for ease in the calculation
    return loss, y_pred

  def updateWeights(self,X,y_pred,y_true,W,alpha,index):
    """
    Parameters:
    X (array) : Independent Features
    y_pred (array) : Predicted Target Variable
    y_true (array) : Dependent Features/ Target Variable
    W (array) : Weights
    alpha (float) : learning rate
    index (int) : Index to fetch the corresponding values of W, X and y 
    Returns:
    W (array) : Update Values of Weight
    """
    for i in range(X.shape[1]):
      #alpha = learning rate, rest of the RHS is derivative of loss function
      W[i] -= (alpha * (y_pred-y_true[index])*X[index][i]) 
    return W

  def train(self, X, y, epochs=10, alpha=0.001, random_state=0):
    """
    Parameters:
    X (array) : Independent Feature
    y (array) : Dependent Features/ Target Variable
    epochs (int) : Number of epochs for training, default value is 10
    alpha (float) : learning rate, default value is 0.001

    Returns:
    y_pred (array) : Predicted Target Variable
    loss (float) : Calculated Sqaured Error Loss for y and y_pred
    """

    num_rows = X.shape[0] #Number of Rows
    num_cols = X.shape[1] #Number of Columns 
    W = np.random.randn(1,num_cols) / np.sqrt(num_rows) #Weight Initialization

    #Calculating Loss and Updating Weights
    train_loss = []
    num_epochs = []
    train_indices = [i for i in range(X.shape[0])]
    for j in range(epochs):
      cost=0
      np.random.seed(random_state)
      np.random.shuffle(train_indices)
      for i in train_indices:
        loss, y_pred = self.forward(X[i],y[i],W[0])
        cost+=loss
        W[0] = self.updateWeights(X,y_pred,y,W[0],alpha,i)
      train_loss.append(cost)
      num_epochs.append(j)
      return W[0], train_loss, num_epochs

  def test(self, X_test, y_test, W_trained):
    """
    Parameters:
    X_test (array) : Independent Features from the Test Set
    y_test (array) : Dependent Features/ Target Variable from the Test Set
    W_trained (array) : Trained Weights
    test_indices (list) : Index to fetch the corresponding values of W_trained,
                          X_test and y_test 

    Returns:
    test_pred (list) : Predicted Target Variable
    test_loss (list) : Calculated Sqaured Error Loss for y and y_pred
    """
    test_pred = []
    test_loss = []
    test_indices = [i for i in range(X_test.shape[0])]
    for i in test_indices:
        loss, y_test_pred = self.forward(X_test[i], W_trained, y_test[i])
        test_pred.append(y_test_pred)
        test_loss.append(loss)
    return test_pred, test_loss

  def plotLoss(self, loss, epochs):
    """
    Parameters:
    loss (list) : Calculated Sqaured Error Loss for y and y_pred
    epochs (list): Number of Epochs

    Returns: None
    Plots a graph of Loss vs Epochs
    """
    plt.plot(epochs, loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Plot Loss')
    plt.show()

if __name__ == "__main__":
  #Get the dataset
  dataset = pd.read_csv(r'weatherHistory.csv')
  print("printing all the columns : ")
  print(dataset.columns)

  # Index(['Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)',
  #        'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
  #        'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover',
  #        'Pressure (millibars)', 'Daily Summary'], dtype='object')

  #Get a glimpse of the Dataset
  dataset.head()

  #Separating the independent and dependent features
  X = np.asarray(dataset['Humidity'].values.tolist()) #Independent feature
  y = np.asarray(dataset['Apparent Temperature (C)'].values.tolist())          #Dependent feature

  #Reshaping the Independent and Dependent features
  X = X.reshape(len(X),1)
  y = y.reshape(len(y),1)

  #Independent Feature Scaling
  X = (X - int(np.mean(X)))/np.std(X)

  #Dependent Feature Scaling
  y = (y - int(np.mean(y)))/np.std(y)

  # print(np.ones((30,1)))

  #Adding the feature X0 = 1, so we have the equation: y =  (W1 * X1) + (W0 * X0) 
  X = np.concatenate((X,np.ones((96453,1))), axis = 1)

  ## Performing Linear Regression
  #Splitting the dataset
  X_train, y_train, X_test, y_test = split_data(X,y)
  #declaring the "regressor" as an object of the class LinearRegression
  regressor = linearRegression()
  #Training 
  W_trained, train_loss, num_epochs = regressor.train(X_train,y_train, epochs=1000, alpha=0.0001)
  #Testing on the Test Dataset
  test_pred, test_loss = regressor.test(X_test, y_test, W_trained)
  # Visualizing Results
  # Plot the Train Loss
  # regressor.plotLoss(train_loss, num_epochs)
  print(f"test_pred : {test_pred}")
  print(f"test_loss : {test_loss}")
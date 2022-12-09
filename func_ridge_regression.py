# Importing libraries
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
  
# Function for model training            
def fit( X, Y, learning_rate, iterations, l2_penality ) :
        
    # no_of_training_examples, no_of_features        
    m, n = X.shape
        
    # weight initialization        
    W = np.zeros( n )
        
    b = 0        
    X = X        
    Y = Y
        
    # gradient descent learning
                
    for i in range( iterations ) :            
        W, b  = update_weights(X, Y, l2_penality, m, learning_rate, W, b)            
    return W, b
      
# Helper function to update weights in gradient descent      
def update_weights( X, Y, l2_penality, m, learning_rate, W, b ) :           
    Y_pred = predict( W, b, X )
        
    # calculate gradients      
    dW = ( - ( 2 * ( X.T ).dot( Y - Y_pred ) ) +               
            ( 2 * l2_penality * W ) ) / m     
    db = - 2 * np.sum( Y - Y_pred ) / m 
        
    # update weights    
    W = W - learning_rate * dW    
    b = b - learning_rate * db        
    return W, b
      
# Hypothetical function  h( x ) 
def predict( W, b, X ) :    
    return X.dot( W ) + b
      
# Driver code
def ridge_regression() :
    # Importing dataset    
    df = pd.read_csv( "weatherHistorysmall.csv" )
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values    
  
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, 
                                            
                                          test_size = 1 / 3, random_state = 0 )
      
    W, b = fit( X_train, Y_train, iterations = 1000, learning_rate = 0.01, l2_penality = 1 )
      
    # Prediction on test set
    Y_pred = predict( W, b, X_test )    
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) )     
    print( "Real values      ", Y_test[:3] )    
    print( "Trained W        ", round( W[0], 2 ) )    
    print( "Trained b        ", round( b, 2 ) )
      
    # Visualization on test set     
    plt.scatter( X_test, Y_test, color = 'blue' )    
    plt.plot( X_test, Y_pred, color = 'orange' )    
    plt.title( 'Humidity vs Apparent Temperature' )
    plt.xlabel( 'Humidity' )
    plt.ylabel( 'Apparent Temperature' )
    plt.show()
      
if __name__ == "__main__" : 
    ridge_regression()

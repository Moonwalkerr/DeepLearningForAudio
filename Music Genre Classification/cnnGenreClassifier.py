import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam
# from tensorflow.python.keras.layers.normalization import BatchNormalization

## Load Data ##
DATA_PATH = '../data.json'
def load_data(data_path):
    
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    
    # converting lists to numpy arrays
    X = np.array(data["mfcc"])
    Y = np.array(data["label"])
    
    return X, Y

def prepare_dataset(test_size,validation_size):
    ### Load Data
    X, Y = load_data(DATA_PATH)
    ### Create Train / test Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=test_size)
    
    ### Create Train / Validation Split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=validation_size)
    
    
    ## Tensorflow for a CNN model accepts 3d Array for each sample
    ### 2d array --> 3d array
    ### (130,13) -->  (130,13,1)  (3rd Dim is the channel / depth) (Audio datas are similar to gray scale images)
    X_train = X_train[...,np.newaxis]   # 4d array --> (num_samples, 130, 13, 1) --> (4000, 13, 13, 1) if 4000 samples
    X_validation = X_validation[...,np.newaxis] 
    X_test = X_test[...,np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test
        


def build_model(input_shape):
    
    ## Create Model
    model = Sequential()
    
    ## 1st Conv layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))      ## 32 = n_filters, (3,3) = grid size
    model.add(MaxPool2D((3,3),  strides=(2,2),padding='same'))  
    ## (3,3) == pool size, padding = zero padding, same across the conv layer
    model.add(BatchNormalization())
    # process that standadizes and normalizes the activation outputs of current layer and to subsequent layer 
    # helps training speed more
    
    ## 2nd Conv layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))     
    model.add(MaxPool2D((3,3),  strides=(2,2),padding='same'))  
    model.add(BatchNormalization())
    
    ## 3rd Conv layer
    model.add(Conv2D(32,(2,2),activation='relu',input_shape=input_shape))  
    model.add(MaxPool2D((2,2),  strides=(2,2),padding='same'))  
    model.add(BatchNormalization())
    
    ## Flatten the output and feed it into dense layer / fully connected layer
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(34,activation='relu'))
    model.add(Dropout(0.3))
    
    
    ## output layer with softmax activation
    model.add(Dense(10,activation='softmax'))
    
    return model

def predict_model(model,X,y):
    
    X = X[np.newaxis,...]  # adding new axis at beginning of the array
    # X = (130,13,1) But model is trained as 4d array (num_samples,130,13,1)
    prediction = model.predict(X)  # X => (1, 130, 13, 1)  4 d array for 1 sample
    # prediction = 2d array [[0.1, 0.4, 0.2, -- , 0.9]] => [[10 diff values for genres]]
    
    # extract the index with max value
    predicted_index = np.argmax(prediction, axis=1)  ## [1] => index of genre label
    
    print("Expected Index : {}, Predicted Index : {}".format(y, predicted_index))


if __name__ == "__main__":
    
    ## Steps ##
    # Create train, validation & test sets
    X_train, X_validation, X_test,  y_train, y_validation, y_test = prepare_dataset(test_size=0.25, validation_size=0.2)
    
    
    ## Input shape 
    input_shape = (X_train.shape[1], X_test.shape[2],X_test.shape[3])   
    # X_Train is 4d array and X_train.shape[0] == num_samples
    
    # Build CNN Architecture (Network)
    model = build_model(input_shape)
    
    # Compile the Network
    optimizer=Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train CNN classifier
    model.fit(X_train, y_train, 
              validation_data=(X_validation,y_validation),
              batch_size=32,epochs=30)
    
    # Evaluate CNN on test sets
    test_error, test_accuracy = model.evaluate(X_test, y_test,verbose=1)
    print("Accuracy on test set is :", test_accuracy)
    
    # make predictions on a sample
    predict_model(model, X_test, y_test)
    
    
    
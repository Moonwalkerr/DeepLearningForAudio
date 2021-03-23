import json
import numpy as np


## Load Data ##
DATA_PATH = '../data.json'
def load_data(data_path):
    
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    
    # converting lists to numpy arrays
    X = np.array(data["mfcc"])
    Y = np.array(data["label"])
    
    return X, Y

if __name__ == "__main__":
    
    ## Steps ##
    # Create train, validation & test sets
    # Build CNN Architecture (Network)
    # Compile the Network
    # Train CNN classifier
    # Evaluate CNN on test sets
    # make predictions on a sample
    
    inputs, targets = load_data(DATA_PATH)
    print(inputs)
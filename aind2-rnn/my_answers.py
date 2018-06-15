import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # get input value and store them in the list
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size])
    
    # get output value and store them in the list 
    y = series[window_size:]
    print(np.shape(X))
    print(np.shape(y))
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    # print out input/output pairs --> here input = X, corresponding output = y
    print ('--- the input X will look like ----')
    print (X)

    print ('--- the associated output y will look like ----')
    print (y)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1))
    
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    from string import punctuation
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    punctuation_allowed = ['!', ',', '.', ':', ';', '?']
    punctuation = list(punctuation)

    punctuation_not_allowed = {x for x in punctuation if x not in punctuation_allowed}
    print(punctuation_not_allowed)

    #text = ''.join(c for c in text if c not in numbers)
    text = ''.join(c for c in text if c not in punctuation_not_allowed)
    text = ''.join(c for c in text if c not in numbers)
    text = text.lower()
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    sliding_range = len(text) - window_size
    for i in range(0, sliding_range, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    #model.add(Dense(num_chars))
    #model.add(Activation = 'softmax')
    
    return model

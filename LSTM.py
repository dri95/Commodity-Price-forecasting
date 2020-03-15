

from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#%% Preprocessing-steps
"""
Importing the dataset(Wheat-US historical prices)
"""
df = pd.read_csv('WUS.csv')
df.head()

"""
Considering only the Prices(time-series data)
"""

df = df['Price'].values
df = df.reshape(-1, 1)
print(df.shape)

"""
Line chart of the Time-series
"""

plt.plot(df)

"""
Splitting the time-series into 80:20 train/test ratio
50 observitions are overlaped for visualization purposes
"""

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8)-50:])
print(dataset_train.shape)
print(dataset_test.shape)

"""
Normalizing the train data in the range(0,1)
"""

scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_train[:5]

"""
Normalizing the test data in the range(0,1)
"""

dataset_test = scaler.transform(dataset_test)
dataset_test[:5]

"""
Function to creating a data structure with 50 time steps and 1 output
"""

def create_datasetlstm(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y
 
"""
Creating x_train and y_train as data structure as 50 time steps and 1 output respectively
"""

x_train, y_train = create_datasetlstm(dataset_train)
x_train[:1]

"""
Creating x_test and y_test as data structure as 50 time steps and 1 output respectively
"""
x_test, y_test = create_datasetlstm(dataset_test)
x_test[:1]

"""
Reshapeing features into a 3D tensor vector for LSTM Layer
"""

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


#%% Model building, parameter selection, Compiling the Network
"""
Building and initializing the Kereas Sequantial Network Model
"""

model = Sequential()

model.add(LSTM(input_shape=(X_train.shape[1], 1),units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

"""
Compliling the Kereas Sequantial Network Model
"""

model.compile(loss='mse', optimizer='adam',metrics=['mean_squared_error', 'mean_absolute_error'])

"""
Summary of the model(displays the units in each layers and number of trainable  weights)
"""

model.summary()


#%% Training the network and PLotting the the train and validation loss
"""
Training the LSTM network model with a validation split of 0.1 for the testing data
"""

history = model.fit(X_train,y_train,batch_size=32,epochs=50,validation_split=0.1)

"""
Plot the Networks train and validation loss
"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss (Adam)')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


#%% Prediction for daily data (one observation into the future)
"""
Predicting one step ahead
"""

predictions1 = model.predict(X_test)

"""
Plotting the one step ahead predictions (full-series plot)
"""

predictions1 = scaler.inverse_transform(predictions1) 
fig, ax = plt.subplots(figsize=(8,4))
plt.plot(df, color='red',  label="True Price")
ax.plot(range(len(y_train)+50,len(y_train)+50+len(predictions1)),predictions1, color='blue', label='Predicted Testing Price')
plt.legend()

"""
Plotting the one step ahead predictions (Predicted-series plot)
"""

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(y_test_scaled, color='red', label='True Testing Price')
plt.plot(predictions1, color='blue', label='Predicted Testing Price')
plt.legend()  


#%% Prediction for multiple days ahead(50 observation into the future)

"""
Function for long term predictions - 50 optimal
"""

def predict_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(len(data)//prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

"""
Predicting for 50 steps ahead
"""
predictionsm = predict_multiple(model, X_test, 50, 50)

"""
Function to plot long term predictions - 50 optimal
"""

def plot_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print ('Done')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

"""
Plotting the 50 steps ahead predictions
"""

plot_multiple(predictionsm, y_test,50)





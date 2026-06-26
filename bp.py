import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
output = np.array([0,1,1,0])

model = Sequential([Dense(2,input_dim=2,activation='sigmoid'),Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(inputs,output,epochs=50)
pred=model.predict(inputs)
print(pred)
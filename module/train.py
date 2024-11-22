import pandas as pd
import numpy as np
import os
from  keras.src.models import Sequential
from  keras.src.layers import Dense
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoding

if __name__ == '__main__':
    MODEL_SAVE_PATH = os.environ['SM_MODEL_DIR']
    INPUT_TRAIN_PATH = os.path.join(os.environ['SM_INPUT_DIR'], 'data/training/train.csv')

    train_data = pd.read_csv(INPUT_TRAIN_PATH, index_column = 0) 
    sklearn.set_config(transform_output="pandas")
    
    le = LabelEncoder()
    train_data["Gender"] = le.fit_transform(df["Gender"])
    train_data["Operating System"] = le.fit_transform(df["Operating System"])
    
    one_hot = OneHotEncoder(sparse_output=False)
    train_data["Device Model"].nunique()
    encoded_cols = one_hot.fit_transform(pd.DataFrame(train_data["Device Model"]))
    train_data = pd.concat([train_data, encoded_cols], axis=1)
    train_data.drop("Device Model", axis=1, inplace=True)
    
    x = train_data.drop("User Behavior Class", axis=1)
    y = train_data["User Behavior Class"]
    
    input_dim = x.shape[1] 
    n_neurons = 250
    epochs = 25

    model = Sequential()

    model.add(Dense(n_neurons, input_dim=input_dim, 
                kernel_initializer='normal', 
                activation='relu'))

    model.add(Dense(5, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, y, epochs=epochs, verbose=1)
    
    model.save(os.path.join(MODEL_SAVE_PATH,'model.keras'))
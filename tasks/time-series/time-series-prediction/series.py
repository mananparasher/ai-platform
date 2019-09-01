# in case this is run outside of conda environment with python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import  unicode_literals

import mlflow
from mlflow import pyfunc
import pandas as pd
import shutil
import tempfile
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import mlflow.tensorflow
import numpy as np
import os
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'








# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

def main(argv):

    zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
    
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    
    
    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)
    
    TRAIN_SPLIT = 300000
    
    
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    uni_data = uni_data.values
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data-uni_train_mean)/uni_train_std
    
    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,univariate_past_history,univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,univariate_past_history,univariate_future_target)
    

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
    
    
    simple_lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),tf.keras.layers.Dense(1)])
    simple_lstm_model.compile(optimizer='adam', loss='mae')
    
    
    with mlflow.start_run():
        mlflow.log_param("Model Summary", simple_lstm_model.summary())
     
        EVALUATION_INTERVAL = 200
        EPOCHS = 10

        simple_lstm_model.fit(train_univariate, epochs=EPOCHS,steps_per_epoch=EVALUATION_INTERVAL,verbose=1)
        
        

        loss=simple_lstm_model.evaluate(val_univariate,steps=100,verbose=1)
        
        mlflow.log_metric("Error", loss)
     
        
      


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)



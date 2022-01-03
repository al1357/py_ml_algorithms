from data_health_insurance import read_data
import pandas as pd
pd.set_option('display.max_columns', 500)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1a) Get processed data
(x, y, data_stats) = read_data()

# 1b) Normalize
x[:, 0] = (x[:, 0] - data_stats["age"]["mean"]) / data_stats["age"]["std"]
x[:, 2] = (x[:, 2] - data_stats["bmi"]["mean"]) / data_stats["bmi"]["std"]
x[:, 3] = (x[:, 3] - data_stats["children"]["mean"]) / data_stats["children"]["std"]

# 2) Train and predict
tf.config.experimental_run_functions_eagerly(True)

# custom error function
def root_mse(y, y_pred):
    y = y.numpy()
    y_pred = y_pred.numpy()
    return np.sqrt(np.sum(np.power((y_pred-y), 2)) / (y.shape[0]))

def build_model():
    model = keras.Sequential([
            #layers.Dense(64, activation=tf.nn.relu, input_shape=[x.shape[1]]),
            layers.Dense(1, input_shape=[x.shape[1]]),
            #layers.Dense(1)
        ])
    
    optimizer = tf.keras.optimizers.SGD(0.03)
    #optimizer = tf.keras.optimizers.RMSprop(0.3)
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', root_mse],
                  run_eagerly=True)
    return model

model = build_model()

model.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
history = model.fit(
    x, y, 
    epochs=300, validation_split=0.2, 
    verbose=0, callbacks=[PrintDot()]
    )


example = x[1330].reshape(1, -1)
example_result = model.predict(example)

print("Predicted result: ", example_result)
print("Real result: ", y[1330])

hist = pd.DataFrame(history.history)
print(hist.head())
print(hist.tail())


def plot_error(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [ins. premium]")
    plt.plot(hist["epoch"], hist["mae"], label="Error")
    plt.plot(hist["epoch"], hist["val_mae"], label="Val Error")    
    plt.legend()
    #plt.ylim([0, 5])
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Square Error [sqrt($ins. premium ^2$)]")
    plt.plot(hist["epoch"], hist["root_mse"], label="Error")
    plt.plot(hist["epoch"], hist["val_root_mse"], label="Val Error")    
    plt.legend()
    #plt.ylim([0, 20])
    
plot_error(history)
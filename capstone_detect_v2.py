# import tkinter GUI
from tkinter import *
from tkinter import ttk # for progress bar
import time
from concurrent.futures import ThreadPoolExecutor

root= Tk()
root.title('Seizure Classifier')

# Import libraries
import numpy as np
import pandas as pd
import datetime
import random
import joblib # to dump pickles
from tensorflow import keras # to load Keras model

# Helper function to peep
import winsound
'''
Helper function to peep once called
'''
def peep_me():
    winsound.Beep(frequency=2500, duration=250)

# input
e1 = Entry(root, width = 35, borderwidth = 5)
e1.grid(row = 0, column = 1, columnspan = 2, padx=10, pady=10)

# output

l1 = Label(root, text='File', bg="white", width = 8, borderwidth = 5)
l1.grid(row = 0, column = 0, columnspan = 1, padx=10, pady=10)

l2 = Label(root, text='True Label', bg="white", width = 8, borderwidth = 5)
l2.grid(row = 1, column = 0, columnspan = 1, padx=10, pady=10)

b1 = Button(root, width = 30, borderwidth = 5)
b1.grid(row = 1, column = 1, columnspan = 2, padx=10, pady=10)

l4 = Label(root, text='Classification', bg="white", width = 8, borderwidth = 10)
l4.grid(row = 2, column = 0, columnspan = 1, padx=10, pady=10)

b2 = Button(root, width = 30, borderwidth = 5)
b2.grid(row = 2, column = 1, columnspan = 2, padx=10, pady=10)


# Progress bar
my_progress = ttk.Progressbar(root, orient=HORIZONTAL,
length=200, mode = 'indeterminate')


# Main Functions

def button_reset_f():
    
    # stop_progress_bar() # stop progress bar

    # current = e.get()
    i = random.randint(10, 100)
    e.delete(0, END)
    e.insert(0, df_seizure.iloc[i, 6])

def button_start_f(df_seizure):
    l1.config(text = 'Time')

    start = int(e1.get())
    # foor loop
    for i in np.arange(start, df_seizure.shape[0]):

        # read e1
        current = e1.get()

        # Predict
        predict(df_seizure, i)

        # increase epoch by one
        # done automatically in the loop

        # Wait 3 screen - adjusted to 2 to allow for predicting delay
        time.sleep(2)

    return

def sec_to_min(sec):
    return  datetime.timedelta(seconds=sec) 

def button_load_file_f():
    start_progress_bar()  # start progress bar
    load_file()
    # peep_me()

    return


def load_file():

    peep_me()
    path = 'data/dataset_2/combined_files/'
    file_name = 'seizure_w_ds2_3s_280321_full.csv'
    executor = ThreadPoolExecutor(1)
    future_file = executor.submit(read_file, path + file_name)

    future_file.add_done_callback(on_file_reading_finished)

    return

# For callback
def on_file_reading_finished(future_file):
    e1.delete(0, END)
    e1.insert(0, 'Loaded, enter starting epoch index.')
    stop_progress_bar() # stop progress bar

# Threading
def read_file(filename):
    e1.delete(0, END)
    e1.insert(0, 'Loading file.....')
    df = pd.read_csv(filename)
    global  df_seizure
    df_seizure = df.loc[:, ~df.columns.str.contains('^Unnamed')] # delete unnamed columns
    peep_me()
    return

def start_progress_bar():
    my_progress.start(10)

def stop_progress_bar():
    my_progress.stop()



# Define Buttons
button_load_file = Button(root, text='Load File', padx=40, pady=20, command=button_load_file_f)
button_reset= Button(root, text='Reset', padx=40, pady=20, command=button_reset_f)
button_start= Button(root, text='Start', padx=40, pady=20, command=lambda: button_start_f(df_seizure))


# Put buttons on the screen
my_progress.grid(row = 3, column = 0, columnspan =3, padx=10, pady=10)
button_load_file.grid(row = 4, column = 0, columnspan = 1, padx=10, pady=10)
button_start.grid(row = 4, column = 1, columnspan = 1, padx=10, pady=10)
button_reset.grid(row = 4, column = 2, columnspan = 1, padx=10, pady=10)



## Deploying a Working Model


# Define function to load a model
def load_joblib_f(path):
    pickle_content = joblib.load(path)
    return pickle_content

# Load model with the function.
joblib_path = 'data/joblib/'
ss_path = joblib_path + 'ss_ictal_050421.joblib'
pca_path = joblib_path + 'pca10_ictal_050421.joblib'
model_path = joblib_path + 'keras/detect_ictal_3b'

# scaler, pca, model = load_joblib_f(ss_path, pca_path, model_path)
scaler = load_joblib_f(ss_path)
pca = load_joblib_f(pca_path)
model = keras.models.load_model(model_path)

def predict(df, index):

    X = df.drop('target', axis = 1)
    X  = X.iloc[index, :]
    X = X.values.reshape(1, -1)
    y = df['target'].values[index]


    # scale
    X_scaled = scaler.transform(X)
    # print(scaler) # debugging

    # pca
    X_pca = pca.transform(X_scaled)

    # Predict
    prediction = model.predict(X_pca)

    if prediction < 0.5:
        pred = 0
    else:
        pred = 1

    # Display
    display(pred, index, y)

    return

def display(pred, index, y):

    # True label
    if y == 0:
        b1.config(bg='green')
    else:
        b1.config(bg='red')

    # Prediction
    if pred == 0:
        b2.config(bg='green')
    else:
        b2.config(bg='red')

    # Epoch number
    e1.delete(0, END)
    e1.insert(0, sec_to_min(int(index) * 3))
    root.update() 

    peep_me()

    print('-------------')                          # verbose
    print(f'Epoch {(index)}.')                    # verbose
    print(f'True label is {round(y,0)}.')      # verbose
    print(f'Prediction is {pred}')           # verbose

    return

root.mainloop()
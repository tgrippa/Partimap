#### Import

# Import libraries
import numpy as np
import os, sys
import pandas as pd
import glob
import re
from tensorflow import keras
import natsort
import sklearn
import tensorflow as tf
import keras_tuner as kt
import datetime
import shutil
import time
import tempfile
import h5py
import matplotlib.pyplot as plt

# Import modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from natsort import natsorted
from keras import backend as K
from keras.utils import np_utils
from keras_tuner import HyperModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add local module to the path
src = os.path.abspath('SRC')
if src not in sys.path:
    sys.path.append(src)
		
# Import functions for processing time information
from processing_time import start_processing, print_processing_time
# Import function that checks and creates folder
from mkdir import check_create_dir
	
# Setup Keras Mixed Precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

#### Create directories 
# Define working path 
root = "/home/ulb/anageo/tgrippa"
output_path = os.path.join(root, "output")
data_path = os.path.join(root, "data")
model_path = os.path.join(output_path, "model")
results_path = os.path.join(output_path, "results")

#Check and create output data directory if needed
list_directories = [output_path, model_path, results_path]
for path in list_directories:
    check_create_dir(path)  

#### Load data
with h5py.File(os.path.join(data_path,"VNIR_128.hdf5"), mode="r") as f:
    x_train = np.asarray(f["x_train"])
    x_test = np.asarray(f["x_test"])
    y_train = np.asarray(f["y_train"])
    y_test = np.asarray(f["y_test"])

#### Model definition
# Specify the R2 calculation formula to use as an assessment metric. 
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Best hyperparameters from tuning
best_hp_A = {'hp_filters_blk1': 32,
			 'hp_filters_blk2': 64,
			 'hp_filters_blk3': 256,
			 'hp_filters_FC1': 128,
			 'hp_filters_FC2': 256,
			 'tuner/epochs': 50,
			 'tuner/initial_epoch': 17,
			 'tuner/bracket': 3,
			 'tuner/round': 3,
			 'tuner/trial_id': '33329bc3ab0971756d77bb78ec740f4f'
			}
best_hp_B = {'hp_bool_blk4': False}
best_hp_C = {'hp_lr': 0.00023988977253075727}

## VGG-like regression model with hyperparameter tuning
def VGG_model(hp):
    model = Sequential()
        
    # block 1 
    hp_filters_blk1 = best_hp_A['hp_filters_blk1'] #Value found after previous Hyperparameter tuning
    model.add(Conv2D(hp_filters_blk1, (3, 3), padding='same', name='block1_conv1', input_shape=input_shape)) #block1_conv1
    model.add(BatchNormalization(axis=-1, name='block1_bn1'))
    model.add(Activation('relu'))
    model.add(Conv2D(hp_filters_blk1, (3, 3), padding='same', name='block1_conv2')) #block1_conv2 
    model.add(BatchNormalization(axis=-1, name='block1_bn2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool')) #block1_pool
    
    # block 2 
    hp_filters_blk2 = best_hp_A['hp_filters_blk2'] #Value found after previous Hyperparameter tuning
    model.add(Conv2D(hp_filters_blk2, (3, 3), padding='same', name='block2_conv1')) #block2_conv1
    model.add(BatchNormalization(axis=-1, name='block2_bn1'))
    model.add(Activation('relu'))
    model.add(Conv2D(hp_filters_blk2, (3, 3), padding='same', name='block2_conv2')) #block2_conv2
    model.add(BatchNormalization(axis=-1, name='block2_bn2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool')) #block2_pool
    
    # block 3 
    hp_filters_blk3 = best_hp_A['hp_filters_blk3'] #Value found after previous Hyperparameter tuning
    model.add(Conv2D(hp_filters_blk3, (3, 3), padding='same', name='block3_conv1')) #block3_conv1
    model.add(BatchNormalization(axis=-1, name = 'block3_bn1'))
    model.add(Activation('relu'))
    model.add(Conv2D(hp_filters_blk3, (3, 3), padding='same', name='block3_conv2')) #block3_conv2
    model.add(BatchNormalization(axis=-1, name='block3_bn2'))
    model.add(Activation('relu'))
    model.add(Conv2D(hp_filters_blk3, (3, 3), padding='same', name='block3_conv3')) #block3_conv3
    model.add(BatchNormalization(axis=-1, name='block3_bn3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block3_pool')) #block3_pool
    
    # block 4 
    hp_bool_blk4 = best_hp_B['hp_bool_blk4'] #Value found after previous Hyperparameter tuning
    if hp_bool_blk4:
        hp_filters_blk4 = best_hp_B['hp_filters_blk4'] #Value found after previous Hyperparameter tuning
        model.add(Conv2D(hp_filters_blk4, (3, 3), padding='same', name='block4_conv1')) #block4_conv1
        model.add(BatchNormalization(axis=-1, name='block4_bn1'))
        model.add(Activation('relu'))
        model.add(Conv2D(hp_filters_blk4, (3, 3), padding='same', name='block4_conv2')) #block4_conv2
        model.add(BatchNormalization(axis=-1, name='block4_bn2'))
        model.add(Activation('relu'))
        model.add(Conv2D(hp_filters_blk4, (3, 3), padding='same', name='block4_conv3')) #block4_conv3
        model.add(BatchNormalization(axis=-1, name='block4_bn3'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block4_pool')) #block4_pool   
    
    # Now to configure the fully conneceted layers 
    
    # FC1
    model.add(Flatten())
    hp_filters_FC1 = best_hp_A['hp_filters_FC1'] #Value found after previous Hyperparameter tuning
    model.add(Dense(hp_filters_FC1, name = 'fc1'))
    model.add(BatchNormalization(axis=-1, name = 'fc1_bn1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1, name = 'fc1_drop1', seed=3)) 
    
    # FC2
    hp_filters_FC2 = best_hp_A['hp_filters_FC2'] #Value found after previous Hyperparameter tuning
    model.add(Dense(hp_filters_FC2, name = 'fc2'))
    model.add(BatchNormalization(axis=-1, name = 'fc2_bn1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1, name = 'fc2_drop1', seed=3))

    # Regression layer 
    model.add(Dense(1, activation = 'linear', name = 'regression'))   
    
    # Optimizer with Keras Tuner on learning rate    
    hp_lr = best_hp_C['hp_lr']
    opt = Adam(learning_rate=hp_lr)
    
    # Compile the model with LOSS, Optimizer, metrics
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mean_absolute_error','RootMeanSquaredError',coeff_determination]) 
   
    return model

# Function for plotting
def plot_loss(history, save_path, show=False):
	fig = plt.subplots(figsize=(15, 9))
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("The loss curve of training and test datasets")
	plt.legend(['train', 'val'], loc='upper left')
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.ylim([0,25.0])
	plt.savefig(save_path, dpi = 300)
	if show:
		plt.show()
	else: 
		plt.close()

def plot_loss_multirun(histories, save_path, show=False):
	loss_list = [x.history["loss"] for x in histories]
	loss_min_bound = [sum(x)/len(x)-np.std(x) for x in zip(*loss_list)]
	loss_max_bound = [sum(x)/len(x)+np.std(x) for x in zip(*loss_list)]
	loss_avg = [sum(x)/len(x) for x in zip(*loss_list)]

	valloss_list = [x.history["val_loss"] for x in histories]
	valloss_min_bound = [sum(x)/len(x)-np.std(x) for x in zip(*valloss_list)]
	valloss_max_bound = [sum(x)/len(x)+np.std(x) for x in zip(*valloss_list)]
	valloss_avg = [sum(x)/len(x) for x in zip(*valloss_list)]

	fig = plt.subplots(figsize=(15, 9))
	plt.plot(loss_avg, color='blue')
	plt.fill_between(np.arange(len(loss_avg)), loss_min_bound, loss_max_bound, facecolor='blue', alpha=0.2)
	plt.plot(valloss_avg, color='orange')
	plt.fill_between(np.arange(len(valloss_avg)), valloss_min_bound, valloss_max_bound, facecolor='orange', alpha=0.3)
	plt.title("The loss curve of training and test datasets")
	plt.legend(['Avg loss train', 'Std loss train','Avg loss test', 'Std loss test'], loc='upper left')
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.ylim([0,25.0])
	plt.savefig(save_path, dpi = 300)
	if show:
		plt.show()
	else: 
		plt.close()

def plot_pred_train(y_train, y_pred_train, R2_train, rmse_train, mae_train, save_path, show=False):
	fig = plt.figure(figsize=(8,8)) 
	ax=plt.subplot(1,1,1) 
	ax.scatter(y_train, y_pred_train)  
	ax.plot(np.linspace(15,35,100), np.linspace(15,35,100), c = 'orange', linestyle='--')
	ax.set_title("Train: prediction vs actual deprivation value")
	ax.set_xlabel("Actual values")
	ax.set_ylabel("Predicted values")
	ax.set_xlim([15,35.0])
	ax.set_ylim([15,35.0])

	ax.text(16, 33, "R-squared = %0.4f" % R2_train, fontsize=12, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left')
	ax.text(16, 32, "RMSE = %0.4f" % rmse_train, fontsize=12, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left')
	ax.text(16, 31, "MAE = %0.4f" % mae_train, fontsize=12, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left')

	plt.savefig(os.path.join(save_path), dpi = 300)
	if show:
		plt.show()
	else: 
		plt.close()

def plot_pred_test(y_train, y_pred_test, R2_test, rmse_test, mae_test, save_path, show=False):
	fig = plt.figure(figsize=(8,8)) 
	ax=plt.subplot(1,1,1) 
	ax.scatter(y_test, y_pred_test)  
	ax.plot(np.linspace(15,35,100), np.linspace(15,35,100), c = 'orange', linestyle='--')
	ax.set_title("Validation dataset: prediction vs actual deprivation value")
	ax.set_xlabel("Actual values")
	ax.set_ylabel("Predicted values")
	ax.set_xlim([15,35.0])
	ax.set_ylim([15,35.0])

	ax.text(16, 33, "R-squared = %0.4f" % R2_test, fontsize=12, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left')
	ax.text(16, 32, "RMSE = %0.4f" % rmse_test, fontsize=12, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left')
	ax.text(16, 31, "MAE = %0.4f" % mae_test, fontsize=12, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left')

	plt.savefig(os.path.join(save_path), dpi = 300)
	if show:
		plt.show()
	else: 
		plt.close()

# Multi_run class
class mutli_run():
    
    def __init__(self, nbrun, earlypatience, modelHP, bsize, nb_epochs, datagen, outputfolder):
        self.nbrun = nbrun
        self.earlypatience = earlypatience
        self.modelHP = modelHP
        self.bsize = bsize
        self.nb_epochs = nb_epochs
        self.datagen = datagen
        self.outputfolder = outputfolder

    def run(self):        
        # Save current time for time management
        starttime = start_processing()
        # Create folder for the multirun 
        check_create_dir(self.outputfolder)
        # Create a table (pandas dataframe) to store the multiple run values
        mutlirun_table = pd.DataFrame(columns = ["Run", "Run_dir", "Weight_file", "RunTime", "Epochs", "R2_train", "rmse_train", "mae_train", "R2_test", "rmse_test", "mae_test"],dtype=object)
        # Create a list of history
        multirun_histories = []
        self.multirun_histories = multirun_histories
        for i in range(1,self.nbrun+1):    
            print("Starting run {}...".format(i))
            # Save current time for time management
            starttime = time.time()
            # Create a new instance of the model
            model = VGG_model(self.modelHP)
            # Define callbacks for this run (needed to reinitialise at least the checkpoint from the previous run)
            checkpoint_filepath = os.path.join(tempfile.mkdtemp(),'Best_performed_model.hdf5')
            checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='min')
            reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=0)
            if self.earlypatience:
                early = EarlyStopping(monitor="val_loss",mode="min",patience=self.earlypatience) # probably needs to be more patient
            tmp_csv_path = os.path.join(tempfile.gettempdir(), "run_log.csv")
            csv_logger = tf.keras.callbacks.CSVLogger(tmp_csv_path, separator=";", append=True)
            callbacklist = [checkpoint, csv_logger]
            if self.earlypatience:
                callbacklist.append(early)
        
            # Train the model 
            history = model.fit(self.datagen.flow(x_train, y_train, batch_size=self.bsize, shuffle=False, seed=3),
                                steps_per_epoch=int(len(x_train)/self.bsize),
                                validation_data = (x_test,y_test),
                                epochs = self.nb_epochs, 
                                callbacks = callbacklist
                                )
            # Save processing time
            training_time = round((time.time() - starttime)/60,1)
            
            # Append the run history to the multirun_hostories list
            multirun_histories.append(history)
            
            # Load the weights from the best checkpoint
            model.load_weights(checkpoint_filepath)
            # run the model on the train and test datasets 
            y_pred_test = model.predict(x_test) 
            y_pred_train = model.predict(x_train)

            # Calculate the R2 Score and RMSE
            R2_train = r2_score(y_train, y_pred_train)
            rmse_train = mean_squared_error(y_train, y_pred_train,squared=False) 
            mae_train = mean_absolute_error(y_train, y_pred_train)
            R2_test = r2_score(y_test,y_pred_test)
            rmse_test = mean_squared_error(y_test, y_pred_test,squared=False) 
            mae_test = mean_absolute_error(y_test, y_pred_test) 

            # Create folder for this run 
            base_name = "val_%0.2f_test_%0.2f" %(R2_train,R2_test)
            runfolder = os.path.join(self.outputfolder, base_name)
            check_create_dir(runfolder)

            # Move tmp LOGGER files to run directory
            shutil.move(tmp_csv_path, os.path.join(runfolder, "run_log.txt"))
            
            # Write metrics on the run log file
            log_path = os.path.join(runfolder, "run_metrics_%s.txt" %base_name)
            content = []
            content.append("the R2 of training dataset is: %0.4f" %R2_train)
            content.append("the RMSE of training dataset is: %0.4f" %rmse_train)
            content.append("the MAE of training dataset is: %0.4f" %mae_train)
            content.append("-----------------------")
            content.append("the R2 of validation dataset is: %0.4f" %R2_test)
            content.append("the RMSE of validation dataset is: %0.4f" %rmse_test)
            content.append("the MAE of validation dataset is: %0.4f" %mae_test)
            with open(log_path, "w") as f:
                for r in content:
                    f.write(r + "\n")

            # Save model weight in the right folder and remote the temporary folder for the ModelCheckpoint
            weight_file = os.path.join(runfolder,'Best_checkoint_%s.hdf5' %base_name)
            shutil.move(checkpoint_filepath, weight_file)
            try:
                shutil.rmtree(os.path.split(checkpoint_filepath)[0])
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            
            # Plots
            plot_loss(history, os.path.join(runfolder,"Loss_MSE_%s.png" %base_name), show=False)
            plot_pred_train(y_train, y_pred_train, R2_train, rmse_train, mae_train, os.path.join(runfolder,"mae_train_%s.png" %base_name), show=False)
            plot_pred_test(y_test, y_pred_test, R2_test, rmse_test, mae_test, os.path.join(runfolder,"mae_test_%s.png" %base_name), show=False)
            
            # Add metrics of the run to the summary table
            run_metrics = pd.Series([i,runfolder,weight_file,training_time,len(history.history['loss']),R2_train,rmse_train,mae_train,R2_test,rmse_test,mae_test], index=mutlirun_table.columns)
            mutlirun_table = mutlirun_table.append(run_metrics, ignore_index=True)
            
        # Create and save the plot loss of multirun
        plot_loss_multirun(mutlirun_instance.multirun_histories, os.path.join(self.outputfolder,'multirun_plot.png'), show=False)
        # Sort the summary table with R2_test descending and save the table to the mutlirun folder
        mutlirun_table.sort_values("R2_test", ascending=False, inplace=True)
        mutlirun_table.reset_index(inplace=False)
        mutlirun_table.to_csv(os.path.join(self.outputfolder,'multirun_table.csv'), index=False)
        self.mutlirun_table = mutlirun_table
        # Compute mean and standard deviation of the runs
        mutlirun_summary = pd.DataFrame()
        mutlirun_summary['Mean'] = mutlirun_table.loc[:, ~mutlirun_table.columns.isin(['Run','Run_dir','Weight_file'])].mean()
        mutlirun_summary['Std'] = mutlirun_table.loc[:, ~mutlirun_table.columns.isin(['Run','Run_di','weight_file'])].std()
        mutlirun_summary.to_csv(os.path.join(self.outputfolder,'multirun_summary.csv'), index=False)
        self.mutlirun_summary = mutlirun_summary
        ## Print processing time
        print_processing_time(starttime ,"Multirun process achieved in ")
		
# Define hyperparameter
input_shape = (128, 128, 4)
bsize = 64
nb_epochs = 50
early_patience = False
datagen = ImageDataGenerator() 

# Train the model multiple times and get summary metrics
#mutlirun_instance = mutli_run(5, early_patience, best_hp_C, bsize, nb_epochs, datagen, os.path.join(results_path,"multirun_500epochs_64bsize_NoEarly"))
mutlirun_instance = mutli_run(5, early_patience, best_hp_C, bsize, nb_epochs, datagen, os.path.join(results_path,"multirun_test"))
mutlirun_instance.run()

#!/usr/bin/env python

"""
Functions for plotting learning curves (loss, val_loss) and scatterplot for regression task
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(history, save_path, ylim=[0,25.0], show=False):
    fig = plt.subplots(figsize=(15, 9))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("The loss curve of training and test datasets")
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.ylim(ylim)
    plt.savefig(save_path, dpi = 300)
    if show:
        plt.show()
    else: 
        plt.close()
        
def plot_loss_multirun(histories, save_path, ylim=[0,25.0], show=False):
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
    plt.ylim(ylim)
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

def plot_pred_test(y_test, y_pred_test, R2_test, rmse_test, mae_test, save_path, show=False):
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

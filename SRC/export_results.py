#!/usr/bin/env python

"""
Export results
"""

import h5py

def save_predictions(save_path,y_pred_test,y_pred_train):
    with h5py.File(save_path, mode="w") as f:
        f["y_pred_test"] = y_pred_test
        f["y_pred_train"] = y_pred_train

def write_run_metrics_file(log_path,R2_train,rmse_train,mae_train,R2_test,rmse_test,mae_test):
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
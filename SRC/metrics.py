#!/usr/bin/env python

"""
Metrics definitions 
"""

def coeff_determination(y_true, y_pred, use_as="metric"):
    '''
    Specify the R2 calculation formula to use as an assessment metric. 
    '''
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    if use_as == "metric":
        return (1 - SS_res/(SS_tot + K.epsilon()))
    if use_as == "loss":
        return (SS_res/(SS_tot + K.epsilon()))
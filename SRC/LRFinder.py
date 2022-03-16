import numpy as np 
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback


class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    Original author: Jeremy Jordan https://gist.github.com/jeremyjordan/ac0229abd4b2b7000aca1643e88e0f02#file-lr_finder-py
    Adapted by Taïs Grippa
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self, figsize=(15, 9), save_path=False, show=True):
        '''Helper function to quickly inspect the learning rate schedule.'''
        fig = plt.subplots(figsize=figsize)
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        if save_path:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()
        else: 
            plt.close()    
            
    def plot_loss(self, figsize=(15, 9), save_path=False, show=True):
        '''Helper function to quickly observe the learning rate experiment results.'''
        fig = plt.subplots(figsize=figsize)
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        if save_path:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()
        else: 
            plt.close()    
                
    def best_lr(self):
        '''Return the learning rate corresponding to the biggest drop in loss.'''
        roc = [x-self.history['loss'][i-1] for i,x in enumerate(self.history['loss']) if i>0]
        roc = [x if not np.isnan(x) else 0 for x in roc]
        return self.history['lr'][np.argmin(roc)+1]
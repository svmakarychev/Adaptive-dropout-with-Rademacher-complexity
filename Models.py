import numpy as np
import matplotlib.pyplot as plt

from matplotlib import image
import os
import time
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
from sklearn.utils import shuffle

from IPython import display

import torch
import torch.nn as nn
import torch.nn.functional as F


def iterate_minibatches(X, y, batchsize):
    indices = np.random.permutation(np.arange(len(X)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]

        
def compute_loss(net, X_batch, y_batch):
    logits = net(X_batch)
    return F.cross_entropy(logits, y_batch).mean()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def Radamacher_Regularization_p_inf_q_1(net, X_batch):
    """
    Calculates p_inf_q_1 Radamacher Regularization for the model,
    discussed in the appendix of the article https://openreview.net/pdf?id=S1uxsye0Z
    Args:
        net: neural network, the last layer should be fC,
                                    with output (number_elements, number_of_classes)
        X_batch (torch.Tensor): Sample matrix, size (batch_size, features)
    
    Return:
        loss (torch.tensor): Radamacher Regularization in the form p_inf_q_ 1
    """
    n, d = X_batch.shape[0], X_batch.shape[1]
    
    k = net[-1].weight.shape[0]
    
    loss = torch.max(torch.abs(X_batch)) * k * np.sqrt(np.log(d) / n)
    
    for layer in net.modules():
        # Take retain probs from VariationalDropout class
        if isinstance(layer, VariationalDropout):
            retain_probability = torch.clamp(layer.probs, 0, 1)
            loss *= torch.sum(torch.abs(retain_probability))
        
        # Take weight from FC layers
        elif isinstance(layer, nn.Linear):
            loss *= 2 * torch.max(torch.abs(layer.weight))
            
            k_new, k_old = layer.weight.shape
            
            loss *= np.sqrt(k_new + k_old) / k_new

    return loss


class VariationalDropout(nn.Module):
    """
    Class for Dropout layer
    Args:
        initial_rates (torch.cuda.tensor): initial points for retain probabilities for
                                            Bernoulli dropout layer
    mode (str): 'deterministic' or 'stochastic'
    """
    def __init__(self, initial_rates, mode):
        super(VariationalDropout, self).__init__()
        
        self.mode = mode
        self.probs = torch.nn.Parameter(initial_rates).cuda()
    
    def forward(self, input):
        
        if self.mode == 'stochastic':
            mask = torch.bernoulli(self.probs.data).view(1, input.shape[1])
        
        elif self.mode == 'deterministic':
            mask = torch.clamp(self.probs, 0, 1).view(1, input.shape[1])
        
        else:
            raise Exception("Check mode: stochastic or deterministic only")
        
        return input * mask


class ComplexModel(object):
    train_loss = []
    train_loss_per_epoch = []
    
    val_loss = []
    val_loss_per_epoch = []

    val_extra_loss = []
    val_extra_loss_per_epoch = []

    val_accuracy = []
    
    def __init__(self, model, X_train, y_train, X_val, y_val, num_epochs, batch_size, optimizer, scheduler, 
                 extra_loss_call, regularizer_weights, random_seed = 123):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.extra_loss_call = extra_loss_call
        self.regularizer_weights = regularizer_weights
        self.random_seed = random_seed
        
    def train(self):
        np.random.seed(self.random_seed)

        self.train_loss = []
        self.train_loss_per_epoch = []

        if self.extra_loss_call:
            self.train_extra_loss = []
            self.train_extra_loss_per_epoch = []


        self.val_loss = []
        self.val_loss_per_epoch = []

        if self.extra_loss_call:
            self.val_extra_loss = []
            self.val_extra_loss_per_epoch = []

        self.val_accuracy = []

        #try:
        for epoch in range(self.num_epochs):
            self.model.train(True) # enable dropout / batch_norm training behavior

            self.scheduler.step()

            for X_batch, y_batch in iterate_minibatches(self.X_train, self.y_train, self.batch_size):
                loss = compute_loss(self.model, X_batch, y_batch)

                if self.extra_loss_call:
                    extra_loss = self.extra_loss_call(self.model, X_batch)
                    self.train_extra_loss.append(float(extra_loss))
                    loss += self.regularizer_weights * extra_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.train_loss.append(float(loss))

            self.train_loss_per_epoch.append(np.mean(self.train_loss[-self.batch_size :]))

            if self.extra_loss_call:
                self.train_extra_loss_per_epoch.append(np.mean(self.train_extra_loss[-self.batch_size :]))

            #########################################
            #       Now lets do validation          #
            #########################################

            self.model.train(False)
            self.val_batch_acc = []
            for X_batch, y_batch in iterate_minibatches(self.X_val, self.y_val, self.batch_size):
                logits = self.model(X_batch)
                y_pred = logits.max(1)[1]
                self.val_batch_acc.append(torch.sum(y_batch == y_pred).cpu().data.numpy() / y_batch.shape[0])

            self.val_accuracy.append(np.mean(np.array(self.val_batch_acc)))

        if self.extra_loss_call:
            return self.train_loss_per_epoch, self.train_extra_loss_per_epoch, self.val_accuracy
        else:
            return self.train_loss_per_epoch, self.val_accuracy
    
    def plot(self):
        # Create some mock data
        objective = np.array(self.train_loss_per_epoch)
        if self.extra_loss_call:
            objective = objective + self.regularizer_weights * np.array(self.train_extra_loss_per_epoch)
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('objective', color=color)
        ax1.plot(objective, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.val_accuracy, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
    def test(self, X_test, y_test):
        self.model.train(False)

        test_batch_acc = []
        for X_batch, y_batch in iterate_minibatches(X_test, y_test, 500):
            logits = self.model(X_batch)
            y_pred = logits.max(1)[1]
            test_batch_acc.append(torch.sum(y_batch == y_pred).cpu().data.numpy() / y_batch.shape[0])

        test_accuracy = np.mean(test_batch_acc)

        print("Final results:")
        print("Test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
        return test_accuracy
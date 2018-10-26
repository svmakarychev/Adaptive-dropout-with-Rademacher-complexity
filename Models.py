import numpy as np
import torch
import torch.nn as nn


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

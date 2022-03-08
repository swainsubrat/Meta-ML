
import utils
import models

from IPython import display
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from l2lutils import KShotLoader
from IPython import display
import torch.nn as nn

class CNP(nn.Module):
    def __init__(self, n_features=1, dims=[32,32], n_ways=5, n_classes=2, lr=1e-4):
        super(CNP, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        dimL1 = [n_features] + dims
        dimL2=[n_features + n_classes*dims[-1]] + dims + [n_classes]
        self.mlp1 = models.MLP(dims=dimL1, task='embedding')
        self.mlp2 = models.MLP(dims=dimL2)
        self.optimizer=torch.optim.Adam(self.parameters(), lr=lr)
    def adapt(self, X, y):
        # INSERT YOUR CODE HERE
        """
        Adaption in CNP is passing each train-task to a MLP(here mlp1)
        and get a hidden representaion(embedding) of it. MLP is equivalent
        to an encoder.
        """
        h = self.mlp1(X)
        idx_0 = y == 0
        idx_1 = y == 1
        avg_0 = torch.mean(h[idx_0], axis=0)
        avg_1 = torch.mean(h[idx_1], axis=0)

        r = torch.cat((avg_0, avg_1))

        return r
        #return r,m,R
    def forward(self, Y, r):
        # INSERT YOUR CODE HERE
        """
        In forward, the task is to pass test-task along with the concatenated
        hidden layers to a MLP(here mlp2) and get the class labels as output.
        """
        r = r.repeat(Y.shape[0], 1)
        x = torch.cat((Y, r), 1)
        
        p = self.mlp2(x)
        return p



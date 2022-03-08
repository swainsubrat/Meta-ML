import utils
import models
import pandas as pd
utils.hide_toggle('Imports 1')

from IPython import display
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from l2lutils import KShotLoader
from IPython import display
utils.hide_toggle('Imports 2')

import learn2learn as l2l
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class MAN(nn.Module):
    def __init__(self, dims=[20,32,32], n_classes=2, lr=1e-3):
        super(MAN,self).__init__()
        self.n_classes = n_classes
        self.mlp = models.MLP(dims=dims,task='embedding')
        self.attn = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self, X, d_train):
        # INSERT YOUR CODE HERE
        x_train, y_train = d_train
        train_embd = self.mlp(x_train)
        test_embd  = self.mlp(X)

        # cosine similarity
        similarities = self.cos(test_embd, train_embd)

        # softmax
        proba = self.attn(similarities)
        one_hot = F.one_hot(y_train, num_classes=self.n_classes)

        preds = []
        for row in proba:
            row = row.repeat(1, one_hot.shape[1]).reshape(-1, one_hot.shape[0])
            row = torch.t(row)
            probas = (row * one_hot).sum(dim=0)
            preds.append(probas)

        preds = torch.stack(preds)

        return preds

    def cos(self, target, ss):
        # compute cosine similarities between 
        # target (batch, embedding_dim) and support set ss(ss_size, embedding_dim)
        # return (batch, ss_size)
        # INSERT YOUR CODE HERE
        batch_size = target.shape[0]
        ss_size    = ss.shape[0]
        similarities = torch.empty(size=(batch_size, ss_size))

        for i in range(batch_size):
            test_sample = target[i].repeat(ss_size, 1)
            similarity = F.cosine_similarity(test_sample, ss, dim=1)
            similarities[i] = similarity

        assert similarities.shape == (batch_size, ss_size)
    
        return similarities

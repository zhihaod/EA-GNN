import torch
import torch.nn as nn
from torch.nn import init
#from torch.autograd import Variable

#import numpy as np
#import time
#import random
#from sklearn.metrics import f1_score,accuracy_score
#from collections import defaultdict

#from encoders import Encoder
#from aggregators import MeanAggregator
#from torch_geometric.datasets import Planetoid

"""
Simple supervised GraphSAGE model 
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes,adj_lists):
        embeds = self.enc(nodes,adj_lists)
        #print(embeds.shape)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes,adj_lists, labels):
        scores = self.forward(nodes,adj_lists)
        return self.xent(scores, labels.squeeze())


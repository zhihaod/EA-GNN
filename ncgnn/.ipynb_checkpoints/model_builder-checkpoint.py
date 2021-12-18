from ncgnn.aggregators import WeightedAggregator,WeightedAggregator_multi
from ncgnn.encoders import Encoder,Encoder_multi
from ncgnn.model import SupervisedGraphSage,SupervisedGraphSage_multi
import torch.nn as nn
import torch

def create_model(features,labels,num_class,num_sample,hidden,cuda,gcn,beta):
    num_nodes = labels.shape[0]
    features_ = nn.Embedding(num_nodes, features.shape[1])
    features_.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
    if cuda:
        features_.cuda()
    agg1 = WeightedAggregator(features_, cuda=cuda,gcn=gcn,beta=beta,subgraph=True,is_first=True)
    enc1 = Encoder(features_, features.shape[1], hidden, aggregator=agg1,num_sample=num_sample, gcn=gcn, cuda=cuda)
    #agg2 = WeightedAggregator_new(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(),cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    #enc2 = Encoder(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), enc1.embed_dim, hidden, aggregator=agg2,num_sample=num_sample_2,base_model=enc1,gcn=True, cuda=cuda)
    graphsage = SupervisedGraphSage(num_class, enc1)   
    if cuda:
        graphsage.cuda()
    return graphsage



def create_model_multi(features,labels,num_class,num_sample,hidden,cuda,gcn,beta):
    num_nodes = labels.shape[0]
    features_ = nn.Embedding(num_nodes, features.shape[1])
    features_.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
    if cuda:
        features_.cuda()
    agg1 = WeightedAggregator_multi(features_, cuda=cuda,gcn=gcn,beta=beta,subgraph=True,is_first=True)
    enc1 = Encoder_multi(features_, features.shape[1], hidden, aggregator=agg1,num_sample=num_sample, gcn=gcn, cuda=cuda)
    graphsage = SupervisedGraphSage_multi(num_class, enc1)   
    if cuda:
        graphsage.cuda()
    return graphsage
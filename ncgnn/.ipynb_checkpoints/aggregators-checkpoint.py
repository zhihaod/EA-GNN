import torch
import torch.nn as nn
from torch.autograd import Variable
import networkx as nx
import random
import time


class WeightedAggregator(nn.Module):
    """
    Aggregates a node's embeddings using ego information
    """
    def __init__(self, features, cuda=False, gcn=False,beta=1,subgraph=True,is_first=True): 
        """
        Initializes the aggregator for a specific graph. 

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(WeightedAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.beta = beta
        self.subgraph = subgraph
        self.is_first = is_first
        
    def forward(self, nodes, adj_lists,to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        nodes = nodes.tolist()
        node_list = nodes.copy()
        _set = set
        if num_sample > 0:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
              
        if self.gcn:
            samp_neighs = [samp_neigh | set([(int(nodes[i]),1+len(adj_lists[nodes[i]]))]) for i, samp_neigh in enumerate(samp_neighs)]
        
        time_start = time.time()
        for samp_neigh in samp_neighs:
            for tur in samp_neigh:
                node_list.append(tur[0])
        time_end = time.time()
        #print(f'time of get nodes:{time_end-time_start}')
        
        unique_nodes_list = list(set(node_list))      
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        
        time_start = time.time()
        if self.gcn:
            for i in range(len(nodes)):
                if (len(samp_neighs[i]) == 1):
                    mask[i,unique_nodes[list(samp_neighs[i])[0][0]]] = 1
                else:
                    for node in samp_neighs[i]: 
                        mask[i,unique_nodes[node[0]]] = pow(node[1],self.beta)
        else:
            for i in range(len(nodes)):
                if (len(samp_neighs[i]) == 0):
                    mask[i,unique_nodes[nodes[i]]] = 1
                else:
                    for node in samp_neighs[i]: 
                        mask[i,unique_nodes[node[0]]] = pow(node[1],self.beta)
        time_end = time.time()
        #print(f'time of getting mask:{time_end-time_start}')
        
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
       

        if self.cuda:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda(),adj_lists)
        else:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list),adj_lists)
        time_start = time.time()
        to_feats = mask.mm(embed_matrix)
        time_end = time.time()
        #print(f'time of getting mask nodes embedding:{time_end-time_start}')
        return to_feats
    
    
    
    
class WeightedAggregator_multi(nn.Module):
    """
    Aggregates a node's embeddings using ego information
    """
    def __init__(self, features, cuda=False, gcn=False,beta=1,subgraph=True,is_first=True): 
        """
        Initializes the aggregator for a specific graph. 
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(WeightedAggregator_multi, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.beta = beta
        self.subgraph = subgraph
        self.is_first = is_first
        
    def forward(self, nodes, adj_lists,to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        nodes = nodes.tolist()
        node_list = nodes.copy()
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
              
        if self.gcn:
            samp_neighs = [samp_neigh | set([(int(nodes[i]),1+len(adj_lists[nodes[i]]))]) for i, samp_neigh in enumerate(samp_neighs)]
        
        for samp_neigh in samp_neighs:
            for tur in samp_neigh:
                node_list.append(tur[0])
      
        unique_nodes_list = list(set(node_list))      
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        mask_list = []
        for _ in range(4):
            mask_list.append(Variable(torch.zeros(len(samp_neighs), len(unique_nodes))))
                
        if self.gcn:
            for i in range(len(nodes)):
                if (len(samp_neighs[i]) == 1):
                    for j in range(len(mask_list)):
                        mask_list[j][i,unique_nodes[list(samp_neighs[i])[0][0]]] = 1
                else:
                    for node in samp_neighs[i]: 
                        for j in range(len(mask_list)):
                            mask_list[j][i,unique_nodes[node[0]]] = pow(node[1],0.5*j)
        else:
            for i in range(len(nodes)):
                if (len(samp_neighs[i]) == 0):
                    for mask in mask_list:
                        mask[i,unique_nodes[nodes[i]]] = 1
                else:
                    for node in samp_neighs[i]: 
                        for i in range(len(mask_list)):
                            mask_list[i][i,unique_nodes[node[0]]] = pow(node[1],0.4*i)
            
                        
        
               
        if self.cuda:
            for mask in mask_list:
                mask = mask.cuda()
       
        mask_list_new = []
        for mask in mask_list:
            num_neigh = mask.sum(1, keepdim=True)
            mask_list_new.append(mask.div(num_neigh).cuda())
     


        if self.cuda:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda(),adj_lists)
        else:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list),adj_lists)
        to_feats = []
        for mask in mask_list_new:
            to_feats.append(mask.mm(embed_matrix))
        to_feats = torch.cat(to_feats,dim=1)
            
        return to_feats
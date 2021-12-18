import torch
import torch.nn as nn
from torch.autograd import Variable
import networkx as nx

import random



    
class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False, is_first=True): 
        """
        Initializes the aggregator for a specific graph. 

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.is_first = is_first
        
    def forward(self, nodes, adj_lists,to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
            
        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
                
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list),adj_lists)
        to_feats = mask.mm(embed_matrix)
        return to_feats



class MeanAggregator2(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False,beta=1,subgraph=True,is_first=True): 
        """
        Initializes the aggregator for a specific graph. 

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator2, self).__init__()

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
        
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
              
        if self.gcn:
            samp_neighs = [samp_neigh | set([int(nodes[i])]) for i, samp_neigh in enumerate(samp_neighs)]
       
        unique_nodes_list = list(set.union(*samp_neighs)) 
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        
        if self.subgraph:
            _subgraph = nx.subgraph
        else:
            _subgraph = nx.ego_graph
            
        G_ = nx.Graph(adj_lists)
        for i in range(len(nodes)):
            G_node = _subgraph(G_,samp_neighs[i])
            #G_node = _subgraph(G_,nodes[i])
            #for n in nx.nodes(G_node):
            for n in list(samp_neighs[i]):
                if len(list(samp_neighs[i])) == 1:
                    mask[i,unique_nodes[n]] = 1
                else:              
                    #mask[i,unique_nodes[n]] = pow(nx.degree(G_node,int(n)),self.beta) 
                    #mask[i,unique_nodes[n]] = pow(nx.degree(G_node,int(n)),(nx.degree(G_node,nodes[i])*(1-nx.clustering(G_node,nodes[i]))))   
                    mask[i,unique_nodes[n]] = pow(nx.degree(G_node,int(n)),(num_sample*(1-nx.clustering(G_node,nodes[i]))))  
                  
                    
        
               
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list),adj_lists)
        to_feats = mask.mm(embed_matrix)
        return to_feats
    
class WeightedAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
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
        
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
              
        #if self.gcn:
            #samp_neighs = [samp_neigh | set([int(nodes[i])]) for i, samp_neigh in enumerate(samp_neighs)]
        samp_neighs = [samp_neigh | set([int(nodes[i])]) for i, samp_neigh in enumerate(samp_neighs)]
        
        unique_nodes_list = list(set.union(*samp_neighs)) 
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        
                
        
        for i in range(len(nodes)):
            n_list = []
            for node in samp_neighs[i]:
                n_list.extend(adj_lists[node] & samp_neighs[i])
            if len(list(samp_neighs[i])) == 1:
                mask[i,unique_nodes[list(samp_neighs[i])[0]]] = 1
            else:                
                for node in samp_neighs[i]:
                    mask[i,unique_nodes[node]] = pow(n_list.count(node),self.beta)
               
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list),adj_lists)
        to_feats = mask.mm(embed_matrix)
        return to_feats
    
class WeightedAggregator_new(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False,beta=1,subgraph=True,is_first=True): 
        """
        Initializes the aggregator for a specific graph. 

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(WeightedAggregator_new, self).__init__()

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
            #samp_neighs = [samp_neigh | set([int(nodes[i])]) for i, samp_neigh in enumerate(samp_neighs)]
            samp_neighs = [samp_neigh | set([(int(nodes[i]),1+len(adj_lists[nodes[i]]))]) for i, samp_neigh in enumerate(samp_neighs)]
        
        
        

        #nodes = nodes.tolist()
        #node_list = nodes.copy()
        for samp_neigh in samp_neighs:
            for tur in samp_neigh:
                node_list.append(tur[0])
      
        unique_nodes_list = list(set(node_list))      
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #print(unique_nodes)
                
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
            
                        
        
               
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
       

        if self.cuda:
            #embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda(),adj_lists)
        else:
            if self.is_first:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list),adj_lists)
        to_feats = mask.mm(embed_matrix)
        return to_feats

    
    
    
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, aggregator,
            num_sample=5,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False,is_first=True): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.is_first = is_first

        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        #self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes,adj_lists):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes,adj_lists,[adj_lists[int(node)] for node in nodes], self.num_sample)
        if not self.gcn:
            if self.cuda:              
                if self.is_first:
                    self_feats = self.features(torch.LongTensor(nodes).cuda())
                else:
                    self_feats = self.features(torch.LongTensor(nodes).cuda(),adj_lists)             
            else:
                if self.is_first:
                    self_feats = self.features(torch.LongTensor(nodes))
                else:
                    self_feats = self.features(torch.LongTensor(nodes),adj_lists)          
            combined = torch.cat([self_feats, neigh_feats], dim=1)    
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined

class Encoder_multi(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, aggregator,
            num_sample=5,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False,is_first=True): 
        super(Encoder_multi, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.is_first = is_first

        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim*4 if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes,adj_lists):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
      
        neigh_feats = self.aggregator.forward(nodes,adj_lists,[adj_lists[int(node)] for node in nodes], self.num_sample)
            
        if not self.gcn:
            if self.cuda:
                #self_feats = self.features(torch.LongTensor(nodes).cuda())
                
                if self.is_first:
                    self_feats = self.features(torch.LongTensor(nodes).cuda())
                else:
                    self_feats = self.features(torch.LongTensor(nodes).cuda(),adj_lists)
            #else:
            #    self_feats = self.features(torch.LongTensor(nodes))
                
            else:
                if self.is_first:
                    self_feats = self.features(torch.LongTensor(nodes))
                else:
                    self_feats = self.features(torch.LongTensor(nodes),adj_lists)
                    
            combined = torch.cat([self_feats, neigh_feats], dim=1)
            
        else:
            combineds = neigh_feats
        
        #combined_list = []
        #for combined in combineds:
        #    combined_list.append(F.relu(self.weight.mm(combined.t())))
        #combined_final = torch.cat(combined_list, dim=0)
        #print(combined.shape)
        return F.relu(self.weight.mm(combineds.t()))
        
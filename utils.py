import numpy as np
from collections import defaultdict
from torch_geometric.datasets import Planetoid
import re



def process_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("./data/cora/raw/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            #feat_data[i,:] = map(float, info[1:-1])
            feat_data[i,:] = info[1:-1]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("./data/cora/raw/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    np.savetxt('./data/cora/feat_data',feat_data,fmt='%s')
    np.savetxt('./data/cora/labels',labels,fmt='%s')
    f = open('./data/cora/adj_lists','w')
    f.write(str(adj_lists))
    f.close() 
    
def process_pubmed():
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("./data/pubmed/raw/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("./data/pubmed/raw/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            
    np.savetxt('./data/pubmed/feat_data',feat_data,fmt='%s')
    np.savetxt('./data/pubmed/labels',labels,fmt='%s')
    f = open('./data/pubmed/adj_lists','w')
    f.write(str(adj_lists))
    f.close() 
    return feat_data,labels,adj_lists
    
    
def process_citeseer():
    dataset = Planetoid(root='./data', name='citeseer')
    
    num_nodes = 3327
    num_feats = 3703
    feat_data = dataset[0].x.numpy()
    labels = np.expand_dims(dataset[0].y.numpy(), axis=1)
    edge_index = dataset[0].edge_index.numpy()
    adj_lists = defaultdict(set)
    
    for i in range(edge_index.shape[1]):
        adj_lists[edge_index[0][i]].add(edge_index[1][i])
        adj_lists[edge_index[1][i]].add(edge_index[0][i])
    for i in range(3327):
        if i not in list(adj_lists.keys()):
            adj_lists[i].add(i)
    np.savetxt('./data/citeseer/feat_data',feat_data,fmt='%s')
    np.savetxt('./data/citeseer/labels',labels,fmt='%s')
    f = open('./data/citeseer/adj_lists','w')
    f.write(str(adj_lists))
    f.close()
    
    
def load_data(dataset,path="./data/"):
    
    full_path = path + dataset
    feat_data = np.loadtxt(full_path+r'/feat_data')
    labels = np.loadtxt(full_path+r'/labels')
    labels = labels.reshape(len(labels),1)
    
    f = open(full_path+r'/adj_lists','r')
    a = f.read()
    p = re.compile(r"^defaultdict\(<class '(\w+)'>")
    c = p.findall(a)[0]
    new_a = a.replace("<class '%s'>"% c, c)
    adj_lists = eval(new_a)
    f.close()
    
    return feat_data,labels,adj_lists
    
    
    
    
    
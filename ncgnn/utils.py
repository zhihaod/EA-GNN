import numpy as np
from collections import defaultdict
from torch_geometric.datasets import *
import re
import networkx as nx
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import random

def shuffle_edges(dict_edges,n_swap,n_try):
    G_ = nx.Graph(dict_edges)
    num_edges = nx.number_of_edges(G_)
    G_swap = nx.double_edge_swap(G_,int(n_swap*num_edges),n_try*int(n_swap*num_edges))
    edge_list = list(nx.to_edgelist(G_swap))
    adj_lists = defaultdict(set)
    for i in range(len(edge_list)):
        adj_lists[edge_list[i][0]].add(edge_list[i][1])
        adj_lists[edge_list[i][1]].add(edge_list[i][0])    
    
    for i in list(set(list(G_.nodes()))-set(list(adj_lists.keys()))): # add single node
        adj_lists[i] = set()
    return adj_lists


def feature_importance_selection(features,ratio,labels,seed):
    rand_clf = RandomForestClassifier(n_jobs=-1)
    rand_clf.fit(features,labels.squeeze())
    importance = rand_clf.feature_importances_
    index = np.argpartition(importance, -int(features.shape[1]*ratio))[-int(features.shape[1]*ratio):]
    feat_data = features[:,index]
    return feat_data

def feature_random_selection(features,ratio,labels,seed):
    random.seed(seed)
    inds = random.sample(range(features.shape[1]),int(features.shape[1]*ratio))
    feat_data = features[:,inds]
    return feat_data

def feature_plain_selection(features,ratio,labels,seed):
    return features[:,:int(features.shape[1]*ratio)]     

def feature_pca_compress(features,ratio,labels,seed):
    pca = PCA(n_components=int(features.shape[1]*ratio))
    feat_data = pca.fit_transform(features)
    return feat_data


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
    
def load_dataset(dataset,path):
    if dataset in ['cora','citeseer','pubmed','academic_cs','academic_py']:
        features,labels,full_adjs_dict =load_data(dataset,path=path)
    elif dataset in ['amazon_computer','amazon_photo']:         
        features,labels,adj_mat =load_data_npz(dataset,path=path)    
        full_adjs_dict = get_adj_dict(adj_mat)
    else:
        try:
            sys.exit(0)
        except:
            print('Dataset does not exist.')    
    return features,labels,full_adjs_dict
    
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
    
    
    
def load_data_npz(dataset,path="./data/"):
    
    loader = np.load(path+f'{dataset}/{dataset}.npz')
    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),shape=loader['adj_shape'])
    
    labels = loader.f.labels
    labels = labels.astype(np.float64)
    labels = labels.reshape(len(labels),1)
    
    feat_data = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),shape=loader['attr_shape']).toarray()
    scaler = StandardScaler()
    scaler.fit(feat_data)
    feat_data = scaler.transform(feat_data)
    
    return feat_data,labels,adj_mat    

def create_data_splits(labels,seed,ratio):
    np.random.seed(seed)
    num_nodes = labels.shape[0]
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes*ratio[0])]
    valid = rand_indices[int(num_nodes*ratio[0]):int(num_nodes*(ratio[0]+ratio[1]))]
    test = rand_indices[int(num_nodes*(ratio[0]+ratio[1])):]
    
    return train,valid,test
    
    
    
#def create_data_splits_mat(labels,adj_mat,seed,ratio):
#    np.random.seed(seed)
#    num_nodes = labels.shape[0]
#    rand_indices = np.random.permutation(num_nodes)
#    train = rand_indices[:int(num_nodes*ratio[0])]
#    valid = rand_indices[int(num_nodes*ratio[0]):int(num_nodes*(ratio[0]+ratio[1]))]
#    test = rand_indices[int(num_nodes*(ratio[0]+ratio[1])):]
#    
#    G_ = nx.from_scipy_sparse_matrix(adj_mat)
#    adj = nx.to_dict_of_lists(G_)
#    for key in adj:
#        adj[key] = set(adj[key])     
#    return train,valid,test,adj

def get_adj_dict(adj_mat):
    G_ = nx.from_scipy_sparse_matrix(adj_mat)
    adj = nx.to_dict_of_lists(G_)
    for key in adj:
        adj[key] = set(adj[key])   
    return adj


def save_list(path,list_):
    file = open(path,'wb')
    pickle.dump(list_,file)
    file.close()
    
    
    
    
    
    
    
def load_data_for_gcn(name,path="./data/"):
    if name in ['cora','citeseer','pubmed']:
        dataset_raw = Planetoid(path+name, name)
    elif 'academic' in name:
        if 'cs' in name:
            dataset_raw = Coauthor(f'{path}{name}', name='CS')
        else:
            dataset_raw = Coauthor(f'{path}{name}', name='Physics')
    elif 'amazon' in name:
        if 'computer' in name:
            dataset_raw = Amazon(f'{path}{name}', name='Computers')
        else:
            dataset_raw = Amazon(f'{path}{name}', name='Photo')
            
    features = dataset_raw[0].x.numpy()
    labels = dataset_raw[0].y.numpy()
    edges = dataset_raw[0].edge_index.numpy()
    
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0,:], edges[1,:])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    

    adj_mat = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    
    return features, labels, adj_mat


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx





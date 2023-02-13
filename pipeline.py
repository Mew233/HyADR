import pickle
from torch.utils.data import DataLoader
import torch
from HNEPY import HNEPY
import numpy as np

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    with open(path, 'rb') as f:
        # obj = pickle.load(f)
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        obj = u.load()
        return obj

def train_test(data_path,output_path,learning_rate,number_epochs):
    d1_fea = load_pkl(data_path + "d1_fea.pkl") #526, 582 (chemical)
    d2_fea = load_pkl(data_path + "d2_fea.pkl") #1702, 1702 (indications)
    d3_fea = load_pkl(data_path + "d3_fea.pkl") # 327, 327

    adj = load_pkl(data_path + "adj.pkl")
    adj = adj.todense() #2555, 2555

    E_pos_train = load_pkl(data_path + "train_E_pos.pkl") # 14, 30
    E_neg_train = load_pkl(data_path + "train_E_neg.pkl")
    E_pos_test = load_pkl(data_path + "test_E_pos.pkl")
    E_neg_test = load_pkl(data_path + "test_E_neg.pkl")


    num_batch = len(E_pos_train)

    d1 = d1_fea.shape[1]
    d2 = d2_fea.shape[1]
    d3 = d3_fea.shape[1]

    K = 3

    r1, r2, r3 = 128, 64, 32
    n1 = len(d1_fea)
    n2 = len(d2_fea)
    n3 = len(d3_fea)

    E_pos_train = torch.from_numpy(np.vstack(E_pos_train))
    E_neg_train = torch.from_numpy(np.vstack(E_neg_train))

    temp_loader_trainval = [E_pos_train,E_neg_train]
    train_val_dataset = torch.utils.data.TensorDataset(*temp_loader_trainval)
    # test_dataset = torch.utils.data.TensorDataset(E_pos_test,E_neg_test)

    trainloader = DataLoader(train_val_dataset, batch_size=10, shuffle=True)

    network = HNEPY(K, d1, d2, d3, n1, n2, n3, r1, r2, r3)
    model = network
    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    for epoch in range(0, number_epochs):
        print(f'Starting epoch {epoch+1}')
        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader):
            model.train()
            pos_edges = data[0]
            neg_edges = data[1]

            # Zero the gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(pos_edges, neg_edges, d1_fea, d2_fea, d3_fea, adj)

            loss = outputs
            loss.backward()
            optimizer.step()
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 9:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 10))
                current_loss = 0.0

    # Evaluation for test set
    print('Training process has finished.')
    print("Starting test")
    
    with torch.no_grad():
        model.eval()
        
        d1_emb = load_pkl("results/" + "d1_emb.pkl") 
        d2_emb = load_pkl("results/" + "d2_emb.pkl") 
        d3_emb = load_pkl("results/" + "d3_emb.pkl") 
        
        # E_pos_train = torch.from_numpy(np.vstack(E_pos_train))
        # E_neg_train = torch.from_numpy(np.vstack(E_neg_train))

        pos_sim, neg_sim = [], []
        for ix, ba in enumerate(zip(E_pos_test,E_neg_test)):
            pos = torch.from_numpy(ba[0])
            neg = torch.from_numpy(ba[1])

            _temp = model.get_bilinear_sim(pos,d1_emb,d2_emb,d3_emb)
            pos_sim.append(_temp)

            _temp_neg = model.get_bilinear_sim(neg,d1_emb,d2_emb,d3_emb)
            neg_sim.append(_temp_neg)
        

        pos_sim = np.reshape(pos_sim, [len(pos_sim)])
        neg_sim = np.reshape(neg_sim, [len(neg_sim)])
        AP, PREC = get_accuracy(pos_sim, neg_sim)

        print("Hyperedge detection by the proposed similarity")
        print("Prec:%.4f" % (PREC))
        print("AP:%.4f" % (AP))

    return loss

def get_accuracy(pos_sim,neg_sim):
    number_positives = len(pos_sim)
    number_negatives = len(neg_sim)
    edge_similarities = np.zeros(number_positives+number_negatives)
    edge_labels = np.zeros(number_positives+number_negatives)
    count = 0
    for i in range(len(pos_sim)):
        edge_similarities[count] = pos_sim[i]
        edge_labels[count] = 1
        count += 1
    for i in range(len(neg_sim)):
        edge_similarities[count] = neg_sim[i]
        edge_labels[count] = 0
        count += 1
    sorted_edge_labels = edge_labels[np.argsort(-edge_similarities)]
    learned_labels = np.zeros(number_positives+number_negatives)
    learned_labels[:number_positives] = 1
    ap = average_precision(sorted_edge_labels)
    prec = precision_at_k(sorted_edge_labels, number_positives)
    return ap,prec

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
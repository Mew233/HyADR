# Inci M. Baytas
# Usuage: python main.py data_path output_path 1e-4 50

import tensorflow as tf
import numpy as np
import sys
import pickle
import argparse


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


from HNE import HNE

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

def get_tuple_sim(E,dim_1,dim_2,dim_3,d1_emb,d2_emb,d3_emb):
    tuple_sim = np.zeros(len(E))
    for i in range(len(E)):
        s = E[i].sum(axis=1)
        edge = E[i][s != 0]
        p = edge[edge[:, 0] == dim_1][:, 1]
        d = edge[edge[:, 0] == dim_2][:, 1]
        a = edge[edge[:, 0] == dim_3][:, 1]
        p_emb = d1_emb[p.astype(int)]
        d_emb = d2_emb[d.astype(int)]
        a_emb = d3_emb[a.astype(int)]
        emb_mat = np.concatenate([p_emb, d_emb, a_emb], axis=0)
        sim = np.matmul(emb_mat, np.transpose(emb_mat))
        diagonal = np.diag(np.diag(sim))
        upper_tri = np.triu(sim)
        sim = upper_tri - diagonal
        sim_vals = sim[np.abs(sim) > 0]
        tuple_sim[i] = sim_vals.mean()
    return tuple_sim

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


def train_test(data_path,output_path,learning_rate,number_epochs):
    d1_fea = load_pkl(data_path + "d1_fea.pkl") #526, 582 (chemical)
    d2_fea = load_pkl(data_path + "d2_fea.pkl") #1702, 1702 (indications)
    d3_fea = load_pkl(data_path + "d3_fea.pkl") # 327, 327 (sda)

    adj = load_pkl(data_path + "adj.pkl")
    adj = adj.todense() #2555, 2555

    E_pos_train = load_pkl(data_path + "train_E_pos.pkl") # 14 (30,255,2)
    E_neg_train = load_pkl(data_path + "train_E_neg.pkl")
    E_pos_test = load_pkl(data_path + "test_E_pos.pkl")# 106 (255,2)
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

    hne = HNE(K, d1, d2, d3, n1, n2, n3, r1, r2, r3)
    cost = hne.get_cost()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # RMSPropOptimizer

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for e in range(number_epochs):
            Cost = np.zeros(num_batch)
            # writer = tf.compat.v1.summary.FileWriter('results/', sess.graph)
            # writer.close()
            # tensorboard --logdir=results/ --host localhost --port 6006
            for i in range(num_batch):
                b1 = sess.run(hne._get_bilinear_sim(E_pos_train[i][0]), feed_dict={hne.pos_edges: E_pos_train[i], \
                                                                 hne.neg_edges: E_neg_train[i], \
                                                                 hne.d1_fea: d1_fea, \
                                                                 hne.d2_fea: d2_fea, \
                                                                 hne.d3_fea: d3_fea,
                                                                 hne.A: adj})
                
                _, Cost[i] = sess.run([optimizer, cost], feed_dict={hne.pos_edges: E_pos_train[i], \
                                                                 hne.neg_edges: E_neg_train[i], \
                                                                 hne.d1_fea: d1_fea, \
                                                                 hne.d2_fea: d2_fea, \
                                                                 hne.d3_fea: d3_fea,
                                                                 hne.A: adj})
            print("Epoch %d, Cost = %.5f" %(e, Cost.mean()))
        print("Training is over!")

        d1_emb, d2_emb, d3_emb = sess.run(hne.get_embs(), feed_dict={hne.pos_edges: E_pos_test, \
                                                                 hne.neg_edges: E_neg_test, \
                                                                 hne.d1_fea: d1_fea, \
                                                                 hne.d2_fea: d2_fea, \
                                                                 hne.d3_fea: d3_fea,
                                                                 hne.A: adj})
        save_pkl(output_path + "d1_emb.pkl", d1_emb)
        save_pkl(output_path + "d2_emb.pkl", d2_emb)
        save_pkl(output_path + "d3_emb.pkl", d3_emb)

        pos_sim, neg_sim = sess.run(hne.edge_similarity(), feed_dict={hne.pos_edges: E_pos_test, \
                                                                hne.neg_edges: E_neg_test, \
                                                                hne.d1_fea: d1_fea, \
                                                                hne.d2_fea: d2_fea, \
                                                                hne.d3_fea: d3_fea,
                                                                hne.A: adj})


        pos_sim = np.reshape(pos_sim, [len(pos_sim)])
        neg_sim = np.reshape(neg_sim, [len(neg_sim)])
        AP, PREC = get_accuracy(pos_sim, neg_sim)

        print("Hyperedge detection by the proposed similarity")
        print("Prec:%.4f" % (PREC))
        print("AP:%.4f" % (AP))

        pos_sim_external = get_tuple_sim(E_pos_test,d1,d2,d3,d1_emb,d2_emb,d3_emb)
        neg_sim_external = get_tuple_sim(E_neg_test, d1, d2, d3, d1_emb, d2_emb, d3_emb)
        AP_external, PREC_external = get_accuracy(pos_sim_external, neg_sim_external)

        print("Hyperedge detection by the baseline similarity computation")
        print("Prec:%.4f" % (PREC_external))
        print("AP:%.4f" % (AP_external))


def main():
    args = arg_parse()
    data_path = args.data_path
    output_path = args.output_path

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    train_test(data_path, output_path, learning_rate, num_epochs)


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--output_path', type=str, default="results/")
    parser.add_argument('--learning_rate', type=float, default=0.004)
    parser.add_argument('--num_epochs', type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    main() 
    #python main.py "data/" "results/" 0.004 10




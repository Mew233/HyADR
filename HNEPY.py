import torch
import torch.nn as nn
from torch import cat,transpose
import pickle

class HNEPY(torch.nn.Module):
    r"""
    Sparse core module with Encoder - Decoder
    """
    def __init__(self, K, d1, d2, d3, n1, n2, n3, r1, r2, r3):
        super(HNEPY, self).__init__()
        
        self.K = K  # Number of types, 3
        self.d1 = d1  # dimensionality of type 1, 582
        self.d2 = d2  # dimensionality of type 2, 1702
        self.d3 = d3  # dimensionality of type 3, 327
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.r1 = r1  # common initial embedding space
        self.r2 = r2  # second layer common embedding dimensionality, 2555
        self.r3 = r3 # common embedding dimensionality

        self.get_embedding_d1 = nn.Sequential(
            nn.Linear(self.d1, self.r1, bias=True),
            nn.Tanh())
        self.get_embedding_d2 = nn.Sequential(
            nn.Linear(self.d2, self.r1, bias=True),
            nn.Tanh())
        self.get_embedding_d3 = nn.Sequential(
            nn.Linear(self.d3, self.r1, bias=True),
            nn.Tanh())
        
        self.GCNN = nn.Sequential(
            nn.Linear(self.r1, self.r2, bias=True),
            nn.Tanh(),
            nn.Linear(self.r2, self.r3, bias=True))
    
        self.get_bilinear_sim = Get_bilinear_sim(self.K, self.r3, self.d1, self.d2, self.d3)
        
    def forward(self, pos_edges, neg_edges, d1_fea, d2_fea, d3_fea, A):
        d1_eb1 = self.get_embedding_d1(torch.from_numpy(d1_fea.astype('float32')))
        d2_eb1 = self.get_embedding_d2(torch.from_numpy(d2_fea.astype('float32')))
        d3_eb1 = self.get_embedding_d3(torch.from_numpy(d3_fea.astype('float32')))
        X = cat([d1_eb1,d2_eb1,d3_eb1],dim=0)
        A = torch.from_numpy(A.astype('float32'))
        emb = self.GCNN(torch.matmul(A,X)) #torch.Size([2555, 32])
        
        d1_eb = emb[:self.n1, :] #[526, 32]
        d2_eb = emb[self.n1:self.n1+self.n2, :] #[1702, 32]
        d3_eb = emb[self.n2:self.n2+self.n3, :] #[327, 32]

        #map function
        # pos_sim = self.get_bilinear_sim(pos_edges,d1_eb,d2_eb,d3_eb) 
        # neg_sim = self.get_bilinear_sim(neg_edges,d1_eb,d2_eb,d3_eb)
        pos_sim, neg_sim = [], []
        for ix, ba in enumerate(zip(pos_edges,neg_edges)):
            pos = ba[0]
            neg = ba[1]
            _temp = self.get_bilinear_sim(pos,d1_eb,d2_eb,d3_eb)
            pos_sim.append(_temp)

            _temp_neg = self.get_bilinear_sim(neg,d1_eb,d2_eb,d3_eb)
            neg_sim.append(_temp_neg)


        Se, Se0 = torch.stack(pos_sim).squeeze(), torch.stack(neg_sim).squeeze()
        sum_Se0 = torch.mean(Se0)
        term1 = torch.mean(torch.log(1 + torch.exp(sum_Se0 - Se)))
        cost = term1

        return cost
    

class Get_bilinear_sim(nn.Module):
    def __init__(self,K, r3, d1, d2, d3):
        super().__init__()
        self.K = K
        self.r3 = r3
        self.d1 = d1  # dimensionality of type 1, 582
        self.d2 = d2  # dimensionality of type 2, 1702
        self.d3 = d3  # dimensionality of type 3, 327

        # self.b1 = nn.Bilinear(self.r3, self.r3, 1, bias=True)
        # self.b2 = nn.Bilinear(self.r3, self.r3, 1, bias=True)
        # self.b3 = nn.Bilinear(self.r3, self.r3, 1, bias=True)
        # tf.matmul(tf.matmul(drug, self.U1), tf.transpose(indi))
        self.b1 = torch.randn(self.r3, self.r3, 1) # bilinear matrix weight (input1 dim, input2 dim, output dim)
        self.b2 = torch.randn(self.r3, self.r3, 1)
        self.b3 = torch.randn(self.r3, self.r3, 1)
                              
        self.B2 = nn.Linear(self.r3, self.K, bias=True)
        # B2 = tf.reduce_mean(tf.matmul(emb_matrix, self.W_lin),0)
        
        self.sim = nn.Linear(self.K, 1, bias=True)
        # sim = tf.matmul(tf.nn.tanh(B + B2 + self.b_lin), self.V)
        
        self.b_lin = torch.randn(self.K)

    def forward(self, indices, d1_eb,d2_eb,d3_eb):
        zero = torch.zeros(len(indices)) #torch.Size([255, 2])
        s = torch.sum(indices, dim=1)
        bool_mask = torch.ne(s, zero)
        indices = indices[bool_mask]
        indices = indices.to(torch.int64)
        emb_indices = indices[:,1]

        dims = indices[:,0]
        bool_d1 = torch.eq(dims, self.d1)
        bool_d2 = torch.eq(dims, self.d2)
        bool_d3 = torch.eq(dims, self.d3)

        d1_indices = torch.masked_select(emb_indices, bool_d1)
        d2_indices = torch.masked_select(emb_indices, bool_d2)
        d3_indices = torch.masked_select(emb_indices, bool_d3)

        drug = d1_eb[d1_indices]
        indi = d2_eb[d2_indices]
        adr = d3_eb[d3_indices]

        emb_matrix = cat([drug,indi,adr],0)
       
        b1 = nn.Parameter(self.b1) #([32, 32, 1])
        b1 = drug @ b1.permute(2,0,1)  #([1, 1, 32]), indi.shape:
        b1 = (b1 @ indi.permute(1,0)).sum(-1).t()  #([4, 1])

        b2 = nn.Parameter(self.b2) 
        b2 = drug @ b2.permute(2,0,1)
        b2 = (b2  @ adr.permute(1,0)).sum(-1).t()

        b3 = nn.Parameter(self.b3)
        b3 = indi @ b3.permute(2,0,1) #([1, 4, 32])
        b3 = b3 @ adr.permute(1,0) #(4, 17)
        b3 = torch.triu(b3, diagonal=1) #表示取矩阵的右上三角，不包括斜对角。
        b3 = torch.reshape(torch.sum(b3),[1,1])

        B = cat([b1,b2,b3],1)
        B2 =  torch.mean(self.B2(emb_matrix),0)
        
        sim = self.sim(torch.tanh(B + B2 + self.b_lin))#
        # sim = tf.matmul(tf.nn.tanh(B + B2 + self.b_lin), self.V)


        #save embedding
        save_pkl("results/" + "d1_emb.pkl", d1_eb)
        save_pkl("results/" + "d2_emb.pkl", d2_eb)
        save_pkl("results/" + "d3_emb.pkl", d3_eb)

        return sim

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)

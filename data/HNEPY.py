# import torch
# import torch.nn as nn
# from torch import cat

# class HNEPY(torch.nn.Module):
#     r"""
#     Sparse core module with Encoder - Decoder
#     """
#     def __init__(self, K, d1, d2, d3, n1, n2, n3, r1, r2, r3):
#         super(HNEPY, self).__init__()
        
#         self.K = K  # Number of types, 3
#         self.d1 = d1  # dimensionality of type 1, 582
#         self.d2 = d2  # dimensionality of type 2, 1702
#         self.d3 = d3  # dimensionality of type 3, 327
#         self.n1 = n1
#         self.n2 = n2
#         self.n3 = n3

#         self.r1 = r1  # common initial embedding space
#         self.r2 = r2  # second layer common embedding dimensionality, 2555
#         self.r3 = r3 # common embedding dimensionality


#         # # Weights of similarity function
#         # self.U1 = self.init_weights(self.r3, self.r3, name='Sim1', reg=None)
#         # self.U2 = self.init_weights(self.r3, self.r3, name='Sim2', reg=None)
#         # self.U3 = self.init_weights(self.r3, self.r3, name='Sim3', reg=None)

#         # self.W_lin = self.init_weights(self.r3, K, name='Sim', reg=None)
#         # self.b_lin = self.init_bias(self.K, name='Sim_bias')
#         # self.V = self.init_weights(self.K, 1, name='Sim_out', reg=None)

#         self.embedding_d1 = nn.Sequential(
#             nn.Linear(self.d1, self.r1, bias=True),
#             nn.Tanh())
#         self.embedding_d2 = nn.Sequential(
#             nn.Linear(self.d2, self.r1, bias=True),
#             nn.Tanh())
#         self.embedding_d3 = nn.Sequential(
#             nn.Linear(self.d3, self.r1, bias=True),
#             nn.Tanh())
        
#         self.GCNN = nn.Sequential(
#             nn.Linear(self.r1, self.r2, bias=True),
#             nn.Tanh(),
#             nn.Linear(self.r2, self.r3, bias=True))
        
#     def get_bilinear_sim(self, indices):
#         zero = torch.zeros
#         s = torch.sum(indices, dim=1)
#         bool_mask = tf.not_equal(s, zero)
#         indices = tf.boolean_mask(indices, bool_mask)
#         emb_indices = tf.slice(indices, [0, 1], [tf.shape(indices)[0], 1])

#         dims = tf.slice(indices, [0, 0], [tf.shape(indices)[0], 1])
#         bool_d1 = tf.equal(dims, self.d1)
#         bool_d2 = tf.equal(dims, self.d2)
#         bool_d3 = tf.equal(dims, self.d3)
#         d1_indices = tf.cast(tf.boolean_mask(emb_indices, bool_d1), tf.int32)
#         d2_indices = tf.cast(tf.boolean_mask(emb_indices, bool_d2), tf.int32)
#         d3_indices = tf.cast(tf.boolean_mask(emb_indices, bool_d3), tf.int32)

#         drug = tf.gather(self.d1_embs, d1_indices)
#         indi = tf.gather(self.d2_embs,d2_indices)
#         adr = tf.gather(self.d3_embs, d3_indices)

#         emb_matrix = tf.concat([drug,indi,adr],0)

#         b1 = tf.matmul(tf.matmul(drug, self.U1), tf.transpose(indi))
#         b1 = tf.reduce_sum(b1)
#         b1 = tf.reshape(b1,[1,1])
#         b2 = tf.matmul(tf.matmul(drug, self.U2), tf.transpose(adr))
#         b2 = tf.reduce_sum(b2)
#         b2 = tf.reshape(b2, [1, 1])
#         b3 = tf.matmul(tf.matmul(indi, self.U3), tf.transpose(adr))
#         b3 = tf.compat.v1.matrix_band_part(b3, 0, -1)
#         b3 = tf.reduce_sum(b3)
#         b3 = tf.reshape(b3, [1, 1])

#         B = tf.concat([b1,b2,b3],1)

#         B2 = tf.reduce_mean(tf.matmul(emb_matrix, self.W_lin),0)


#         sim = tf.matmul(tf.nn.tanh(B + B2 + self.b_lin), self.V)

#         return sim

#     def forward(self, pos_edges, neg_edges, d1_fea, d2_fea, d3_fea, A):
#         d1_eb1 = self.get_embedding_d1(d1_fea)
#         d2_eb1 = self.get_embedding_d2(d2_fea)
#         d3_eb1 = self.get_embedding_d3(d3_fea)
#         X = cat([d1_eb1,d2_eb1,d3_eb1],dim=0)
#         emb = self.GCNN(A,X)
#         d1_eb = tf.slice(emb, [0,0], [self.n1, self.r3])
#         d2_eb = tf.slice(emb,[self.n1,0],[self.n2,self.r3])
#         d3_eb = tf.slice(emb, [(self.n1 + self.n2), 0], [self.n3, self.r3])

#         nn.Bilinear(20, 30, 40)
#         pos_sim = self.get_bilinear_sim(pos_edges) #
#         neg_sim = self.get_bilinear_sim(neg_edges)

#         Se, Se0 = pos_sim, neg_sim
#         sum_Se0 = tf.reduce_mean(Se0)
#         term1 = tf.reduce_mean(tf.compat.v1.log(1 + tf.compat.v1.exp(sum_Se0 - Se)))
#         cost = term1

#         return cost
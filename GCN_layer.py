import tensorflow as tf
import numpy as np


def dot(A, B, sparse):
    if sparse:
        dot_res = tf.sparse.sparse_dense_matmul(A, B)
    else:
        dot_res = tf.matmul(A, B)
    return dot_res


class GCN_layer(tf.keras.Model):
    def __init__(self, hid_dim_list, featureless, layer_num, activation, **kwargs):
        # hid_dim_list: size = layer_num+1,  hid[0]=feat_dim, hid[-1]=gcn_output
        super(GCN_layer, self).__init__(**kwargs)
        self.weightlist = []
        self.layer_num = layer_num
        for i in range(self.layer_num):
            self.weightlist.append(
                self.add_weight('weight' + str(i), [hid_dim_list[i], hid_dim_list[i + 1]], initializer='random_normal'))
        self.featureless = featureless
        self.activation = activation

    def __call__(self, inputs, **kwargs):
        # for one timestamp
        # inputs: [adj,x]
        H_n = []
        for i in range(self.layer_num):
            if i == 0:
                if self.featureless:
                    h = self.weightlist[i]
                else:
                    h = dot(inputs[1], self.weightlist[i], sparse=False)
            else:
                # H(t) = H(t-1)*W[t]
                h = dot(H_n[-1], self.weightlist[i], sparse=False)
                H_n = []
            H_n.append(dot(inputs[0], h, sparse=False))
        return self.activation(H_n[-1])


class GRCU(tf.keras.Model):
    def __init__(self, args, activation, featureless):
        super(GRCU, self).__init__()

        self.evolve_weights = mat_GRU_cell(args)

        self.activation = activation
        self.featureless = featureless
        self.GCN_init_weights = self.add_weight('GCN_init_weight', [args[0], args[1]], initializer='random_normal')
        # self.reset_param(self.GCN_init_weights)

    # def reset_param(self, t):
    #     # Initialize based on the number of columns
    #     stdv = 1. / tf.sqrt(t.size(1))
    #     t.data.uniform_(-stdv, stdv)

    def __call__(self, inputs, **kwargs):  # ,mask_list):
        if self.featureless:
            A_list = inputs
        else:
            A_list, node_embs_list = inputs
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list[0]):
            if not self.featureless:
                node_embs = node_embs_list[t]
                # first evolve the weights from the initial and use the new weights with the node_embs
                GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t])
                node_embs = self.activation(tf.matmul(Ahat,tf.matmul(node_embs,GCN_weights)))
            else:
                GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t])
                node_embs = self.activation(tf.matmul(Ahat, GCN_weights))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # args: rows,cols
        self.update = mat_GRU_gate(args[0],
                                   args[1],
                                   tf.nn.sigmoid)

        self.reset = mat_GRU_gate(args[0],
                                  args[1],
                                  tf.nn.sigmoid)

        self.htilda = mat_GRU_gate(args[0],
                                   args[1],
                                   tf.nn.tanh)

        # self.choose_topk = TopK(feats=args[0],
        #                         k=args[1])

    def __call__(self, prev_Q, **kwargs):  # ,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update((z_topk, prev_Q))
        reset = self.reset((z_topk, prev_Q))

        h_cap = reset * prev_Q
        h_cap = self.htilda((z_topk, h_cap))

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(tf.keras.Model):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = self.add_weight('W', shape=[rows, rows], initializer='random_normal')
        # self.reset_param(self.W)

        self.U = self.add_weight('u', shape=[rows, rows], initializer='random_normal')
        # self.reset_param(self.U)

        self.bias = tf.zeros(shape=[rows, cols])
        self.rows = rows

    def reset_param(self, t, rows):
        # Initialize based on the number of columns
        # print(rows)
        stdv = 1 / np.sqrt(rows)
        return tf.random.uniform(shape=t.get_shape(), minval=-stdv, maxval=stdv)

    def __call__(self, inputs, **kwargs):
        x, hidden = inputs
        self.W = self.reset_param(self.W, self.rows)
        self.U = self.reset_param(self.U, self.rows)
        out = self.activation(tf.matmul(self.W, x) + \
                              tf.matmul(self.U, hidden) + \
                              self.bias)
        return out


#
#
# class TopK(tf.keras.Model):
#     def __init__(self, feats, k):
#         super().__init__()
#         self.scorer = Parameter(torch.Tensor(feats, 1))
#         self.reset_param(self.scorer)
#
#         self.k = k
#
#     def reset_param(self, t):
#         # Initialize based on the number of rows
#         stdv = 1. / tf.sqrt(t.size(0))
#         t.data.uniform_(-stdv, stdv)
#
#     def forward(self, node_embs, mask):
#         scores = node_embs.matmul(self.scorer) / self.scorer.norm()
#         scores = scores + mask
#
#         vals, topk_indices = scores.view(-1).topk(self.k)
#         topk_indices = topk_indices[vals > -float("Inf")]
#
#         if topk_indices.size(0) < self.k:
#             topk_indices = u.pad_with_last_val(topk_indices, self.k)
#
#         tanh = tf.nn.tanh()
#
#         if isinstance(node_embs, torch.sparse.FloatTensor) or \
#                 isinstance(node_embs, torch.cuda.sparse.FloatTensor):
#             node_embs = node_embs.to_dense()
#
#         out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))
#
#         # we need to transpose the output
#         return out.t()
#
def mask(metric):
    m_one = tf.ones_like(metric)
    m_zero = tf.zeros_like(metric)
    metric_onehot = tf.where(metric > 0, x=m_one, y=m_zero)

    return mask

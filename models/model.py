import torch
from torch import nn

from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer, PatientFeatureLayer
from models.utils import DotProductAttention


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)    # (batch_size, hidden_size)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj,  # 这里传入的 adj 实际上是 tuple (adj_stat, adj_ont)
                 graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation,
                 feature_input_dim, device):
        super().__init__()

        # 解包 adj (因为 load_adj 返回的是 tuple)
        # 如果传进来的是单个 Tensor (旧代码)，则做兼容
        if isinstance(adj, tuple) or isinstance(adj, list):
            self.adj_stat = adj[0]
            self.adj_ont = adj[1]
        else:
            self.adj_stat = adj
            self.adj_ont = adj

        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)

        # 初始化 GraphLayer (保持原样调用，参数内部兼容)
        self.graph_layer = GraphLayer(self.adj_stat, code_size, graph_size)

        self.patient_feature_layer = PatientFeatureLayer(feature_input_dim, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size,
                                                feature_input_dim)
        self.attention = DotProductAttention(hidden_size, 32)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.device = device

    def forward(self, code_x, divided, neighbors, lens, pid_index, data):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        #-----新添加-----

        for code_x_i, divided_i, neighbor_i, len_i, pid_index_i in zip(code_x, divided, neighbors, lens, pid_index):
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            patinet_tensor = torch.tensor(data[pid_index_i]).to(self.device)

            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                # [关键修改] 显式传入两个图给 GraphLayer
                co_embeddings, no_embeddings = self.graph_layer(
                    code_x=c_it,
                    neighbor=n_it,
                    c_embeddings=c_embeddings,
                    n_embeddings=n_embeddings,
                    adj_stat=self.adj_stat,
                    adj_ont=self.adj_ont
                )

                patient_data = patinet_tensor[len_it]
                similarity_matrix = self.patient_feature_layer(patient_data, co_embeddings)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings,
                                                       similarity_matrix, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)

            output_i = self.attention(torch.vstack(output_i))
            output.append(output_i)

        output = torch.vstack(output)
        output = self.classifier(output)
        return output
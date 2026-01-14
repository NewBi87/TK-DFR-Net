import torch
from torch import nn

from models.ablation_layer import EmbeddingLayer, GraphLayer, TransitionLayer, PatientFeatureLayer
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
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation,
                 feature_input_dim, device):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        # 特征级层初始化
        self.patient_feature_layer = PatientFeatureLayer(feature_input_dim, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size, feature_input_dim)
        self.attention = DotProductAttention(hidden_size, 32)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)

        self.device = device

    def forward(self, code_x, divided, neighbors, lens, pid_index, data):
        embeddings = self.embedding_layer()
        #c_embeddings, n_embeddings, u_embeddings = embeddings
        
        output = []
        for code_x_i, divided_i, neighbor_i, len_i, pid_index_i in zip(code_x, divided, neighbors, lens, pid_index):    #遍历每个样本
            no_embeddings_i_prev = None # 存储前一个时间步的节点嵌入
            output_i = []   # 收集每个时间步的输出
            h_t = None  # 隐藏状态，用于在每个时间步之间传递信息
            patinet_tensor = torch.tensor(data[pid_index_i]).to(self.device)
            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                co_embeddings = self.graph_layer(c_it, None, embeddings)     # 诊断级图结构处理
                patient_data = patinet_tensor[len_it]
                similarity_matrix = self.patient_feature_layer(patient_data, co_embeddings)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, None, similarity_matrix, h_t)
                # output_it, h_t, c_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, similarity_matrix, h_t, c_t)
                # no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            output_i = self.attention(torch.vstack(output_i))
            output.append(output_i)

        output = torch.vstack(output)       # 将一个包含多个张量的列表堆叠成一个单一的张量，形状为[batch_size, hidden_size]
        output = self.classifier(output)
        return output

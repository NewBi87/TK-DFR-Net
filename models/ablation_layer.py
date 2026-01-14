import torch
from torch import nn
from models.utils import SingleHeadAttentionLayer, AttentionFusionLayer, FusionLayer, CrossModalAttention, WeightedSumFusionLayer
import torch.nn.functional as F

# 生成三种不同嵌入矩阵的层
class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size):
        super().__init__()
        # 仅保留单一通用嵌入矩阵
        self.universal_embeddings = nn.Parameter(
            data=nn.init.xavier_uniform_(torch.empty(code_num, code_size))
        )

    def forward(self):
        return self.universal_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj  # 全局静态图A
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, universal_embeddings):
        """
        输入变化：
        - 去除了动态子图划分，仅使用全局图A
        - code_x 仍表示当前诊断的multi-hot向量
        """
        # 生成通用节点嵌入
        node_embeddings = universal_embeddings

        # 全局图卷积（公式5、6的静态版本）
        aggregated = torch.matmul(self.adj, node_embeddings)
        combined = node_embeddings + aggregated

        # 统一处理所有节点类型
        graph_output = self.activation(self.dense(combined))
        return graph_output

# 消融
class PatientFeatureLayer(nn.Module):
    def __init__(self, feature_input_dim, graph_size):
        super().__init__()
        # 特征值投影
        self.value_proj = nn.Linear(1, graph_size)
        # 特征重要性门控
        self.gate = nn.Sequential(
            nn.Linear(feature_input_dim, feature_input_dim),
            nn.Sigmoid()
        )
        self.cosine_sim = nn.CosineSimilarity(dim=2)

    def forward(self, data_tensor, co_embeddings):

        # 直接使用原始数据均值）
        # data_tensor(seq_len, feature_input_dim)
        # co_embeddings(code_num, graph_size)
        global_feat = data_tensor.mean(dim=0)   #(feature_input_dim,)

        # 特征值投影（每个特征单独编码）
        feat_values = global_feat.unsqueeze(-1) #(feature_input_dim,1)
        project_feat = self.value_proj(feat_values) #(feature_input_dim, graph_size)


        # 疾病-特征相关性计算
        # corr_scores = torch.matmul(
        #     co_embeddings,
        #     project_feat.T
        # )
        corr_scores = self.cosine_sim( #(code_num, feature_input_dim)
            co_embeddings.unsqueeze(1),
            project_feat.unsqueeze(0)
        )

        # 动态特征门控
        gate_weights = self.gate(global_feat)  # (embed_size,)
        similarity = corr_scores * gate_weights.unsqueeze(0)
        return similarity   #(code_num, feature_input_dim)




class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size, feature_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)  # GRU
        # 单头注意力机制，用于处理不同类型编码之间的交互。输入维度为 graph_size，注意力机制的中间层维度为 t_attention_size，输出维度为 t_output_size。
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.cross_modal_attention = CrossModalAttention(graph_size, graph_size, t_output_size, t_attention_size)
        self.feature_to_graph = nn.Linear(feature_size, graph_size)
        # self.fusion_layer = FusionLayer(hidden_size)
        self.fusion_layer = WeightedSumFusionLayer(hidden_size)
        self.embeddings_fusion_layer = WeightedSumFusionLayer(graph_size)
        self.embedding_fusion = nn.Linear(2 * graph_size, graph_size)
        self.linear_fusion = nn.Linear(2 * hidden_size, hidden_size)
        self.layer_norm_sim = nn.LayerNorm(graph_size)
        self.layer_norm_co = nn.LayerNorm(graph_size)
        self.layer_norm_hd = nn.LayerNorm(hidden_size)
        self.layer_norm_hz = nn.LayerNorm(hidden_size)

        self.activation = nn.Tanh()

        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, similarity_matrix,
                hidden_state=None):
        # 分割索引`
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
        m1_index = torch.where(m1 > 0)[0]  # m1中大于0，表示共现编码的索引
        m2_index = torch.where(m2 > 0)[0]  # m2中大于0，表示邻居编码的索引
        m3_index = torch.where(m3 > 0)[0]  # m3中大于0，表示无关编码的索引
        # 初始化隐藏状态
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        output_m1 = 0
        output_m23 = 0

        no_embeddings = co_embeddings
        unrelated_embeddings = co_embeddings
        # 处理共现诊断
        if len(m1_index) > 0:
            # 原方法
            m1_embedding = co_embeddings[m1_index]
            h = hidden_state[m1_index] if hidden_state is not None else None  # hidden_state不为None则赋值，即非第一时间步
            h_m1 = self.gru(m1_embedding, h)

            h_new[m1_index] = h_m1
            output_m1, _ = torch.max(h_m1, dim=-2)  # 最大池化

        # 处理新出现的邻居m2和无关诊断m3
        if t > 0 and len(m2_index) + len(m3_index) > 0:

            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])  # q
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])  # v
            # h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_d = self.single_head_attention(q, q, v)

            similarity_embeddings = self.feature_to_graph(similarity_matrix)
            similarity_embeddings_normalized = self.layer_norm_sim(similarity_embeddings)
            co_embeddings_normalized = self.layer_norm_co(co_embeddings)

            # 线性融合方法：拼接、降维
            # fusion_embeddings = torch.cat([similarity_embeddings_normalized, co_embeddings_normalized], dim=-1)
            # fo_embeddings = self.embedding_fusion(fusion_embeddings)

            # 其他融合方法
            fo_embeddings = self.embeddings_fusion_layer(similarity_embeddings_normalized, co_embeddings_normalized)


            # 结合特征学习
            q_z = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
            k_z = torch.vstack([fo_embeddings[m2_index], fo_embeddings[m3_index]])
            v_z = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
            h_z = self.cross_modal_attention(q_z, k_z, v_z)

            # h_d_normalized = self.layer_norm_hd(h_d)
            # h_z_normalized = self.layer_norm_hz(h_z)
            # h_m23 = self.fusion_layer(h_d_normalized, h_z_normalized)

            #h_m23 = self.fusion_layer(h_d, h_z)
            h_m23 = self.activation(self.fusion_layer(h_d, h_z))
            #h_m23 = self.activation(self.attention_fusion_layer(h_d, h_z))
            #h_m23 = self.fusion_layer(h_d, h_z, v_z)
            #h_m23 = self.linear_fusion(torch.cat([h_d, h_z], dim=-1))

            # h_m23的前半部分分给m2,后半分给m3_index
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]

            output_m23, _ = torch.max(h_m23, dim=-2)  # 计算h_m23最大值

        if len(m1_index) == 0:
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:
            output = output_m1
        else:
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new


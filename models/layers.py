import torch
from torch import nn
from models.utils import SingleHeadAttentionLayer, AttentionFusionLayer, FusionLayer, CrossModalAttention, WeightedSumFusionLayer
import torch.nn.functional as F

# 生成三种不同嵌入矩阵的层
class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


# models/layers.py -> GraphLayer (Gated Version)

class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense_stat = nn.Linear(code_size, graph_size)
        self.dense_ont = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

        # [修改 1] 定义融合层 (计算 Correction)
        self.fusion_linear = nn.Linear(graph_size * 2, graph_size)

        # [新增] 定义门控层 (计算 Gate)
        # 输入是拼接特征，输出是 0~1 的门控系数
        self.gate_linear = nn.Linear(graph_size * 2, graph_size)

        # [关键] ★★★ 零初始化 (双重保险) ★★★
        # 1. 修正量 Correction 初始化为 0 (保证开局不崩)
        nn.init.zeros_(self.fusion_linear.weight)
        nn.init.zeros_(self.fusion_linear.bias)

        # 2. 门控 Gate 初始化
        # 我们可以让门控初始偏置为负 (如 -2.0)，这样初始 sigmoid(gate) 接近 0
        # 这意味着模型开局几乎就是纯单图模型，随着训练逐渐打开门控
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, -2.0)

    # 原融合：确诊疾病残差融合，邻居疾病直接输出
    def forward(self, code_x, neighbor, c_embeddings, n_embeddings, adj_stat=None, adj_ont=None):
        if adj_stat is None: adj_stat = self.adj
        if adj_ont is None: adj_ont = self.adj

        # ... (特征提取部分不变) ...
        center_codes = torch.unsqueeze(code_x, dim=-1)
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)

        center_embed_stat = center_codes * c_embeddings
        agg_stat = torch.matmul(adj_stat, center_embed_stat)
        h_stat = self.activation(self.dense_stat(center_embed_stat + agg_stat))

        center_embed_ont = center_codes * c_embeddings
        agg_ont = torch.matmul(adj_ont, center_embed_ont)
        h_ont = self.activation(self.dense_ont(center_embed_ont + agg_ont))

        # [核心升级] 门控残差拼接 (Gated Residual Concatenation)
        combined = torch.cat([h_stat, h_ont], dim=-1)

        # 1. 计算修正量 (Correction)
        correction = self.activation(self.fusion_linear(combined))

        # 2. 计算门控 (Gate) - 范围 (0, 1)
        gate = torch.sigmoid(self.gate_linear(combined))

        # 3. 门控残差连接
        # H = H_stat + Gate * Correction
        # 简单样本 Gate -> 0, 复杂样本 Gate -> 1
        h_fused = h_stat + gate * correction

        # ... (邻居处理不变) ...
        neighbor_embed = neighbor_codes * n_embeddings
        agg_neighbor = torch.matmul(adj_stat, neighbor_embed)
        no_embeddings = self.activation(self.dense_stat(neighbor_embed + agg_neighbor))

        return h_fused, no_embeddings

    # def forward(self, code_x, neighbor, c_embeddings, n_embeddings, adj_stat=None, adj_ont=None):
    #     if adj_stat is None: adj_stat = self.adj
    #     if adj_ont is None: adj_ont = self.adj
    #
    #     # ==========================================
    #     # 1. 中心确诊疾病 (Target Codes) 的双图融合
    #     # ==========================================
    #     center_codes = torch.unsqueeze(code_x, dim=-1)
    #     center_embed_stat = center_codes * c_embeddings
    #     agg_stat = torch.matmul(adj_stat, center_embed_stat)
    #     h_stat = self.activation(self.dense_stat(center_embed_stat + agg_stat))
    #
    #     center_embed_ont = center_codes * c_embeddings
    #     agg_ont = torch.matmul(adj_ont, center_embed_ont)
    #     h_ont = self.activation(self.dense_ont(center_embed_ont + agg_ont))
    #
    #     combined_c = torch.cat([h_stat, h_ont], dim=-1)
    #     correction_c = self.activation(self.fusion_linear(combined_c))
    #     gate_c = torch.sigmoid(self.gate_linear(combined_c))
    #     h_fused = h_stat + gate_c * correction_c
    #
    #     # ==========================================
    #     # 2. 邻居历史疾病 (Neighbor Codes) 的对称双图融合 [本次修改核心]
    #     # ==========================================
    #     neighbor_codes = torch.unsqueeze(neighbor, dim=-1)
    #     neighbor_embed = neighbor_codes * n_embeddings
    #
    #     # (a) 统计图聚合
    #     agg_neighbor_stat = torch.matmul(adj_stat, neighbor_embed)
    #     no_stat = self.activation(self.dense_stat(neighbor_embed + agg_neighbor_stat))
    #
    #     # (b) 本体图聚合
    #     agg_neighbor_ont = torch.matmul(adj_ont, neighbor_embed)
    #     no_ont = self.activation(self.dense_ont(neighbor_embed + agg_neighbor_ont))
    #
    #     # (c) 复用门控机制进行融合 (Weight Sharing)
    #     combined_n = torch.cat([no_stat, no_ont], dim=-1)
    #     correction_n = self.activation(self.fusion_linear(combined_n))
    #     gate_n = torch.sigmoid(self.gate_linear(combined_n))
    #
    #     # (d) 门控残差输出
    #     no_fused = no_stat + gate_n * correction_n
    #
    #     return h_fused, no_fused


# class PatientFeatureLayer(nn.Module):
#     def __init__(self, feature_input_dim, graph_size):
#         super().__init__()
#         #self.rnn_hidden_dim = rnn_hidden_dim
#         self.cosine_similarity = nn.CosineSimilarity(dim=-1)
#         self.feature_input_dim = feature_input_dim
#         #self.rnn = nn.GRU(self.feature_input_dim, rnn_hidden_dim, batch_first=True)
#         self.linear = nn.Linear(1, graph_size)
#         self.activation = nn.ReLU()
#         #self.embedding_layer = nn.Linear(rnn_hidden_dim, graph_size)
#
#     def forward(self, data_tensor, co_embeddings):
#         feature_input = self.activation(self.linear(data_tensor.T.unsqueeze(-1)))
#         similarity = self.cosine_similarity(feature_input.mean(dim=1), co_embeddings.unsqueeze(1))
#         return similarity

# 消融实验
# class PatientFeatureLayer(nn.Module):
#     def __init__(self, feature_input_dim, graph_size):
#         super().__init__()
#         self.cosine_similarity = nn.CosineSimilarity(dim=-1)
#         self.linear = nn.Linear(1, graph_size)
#         self.activation = nn.ReLU()
#
#     def forward(self, data_tensor, co_embeddings):
#         # 输入处理与原始模型一致
#         feature_input = self.activation(self.linear(data_tensor.T.unsqueeze(-1)))  # [seq_len, batch, graph_size]
#
#         # 将均值池化替换为最大池化
#         aggregated_features = feature_input.max(dim=1).values  # [batch, graph_size]
#
#         # 计算相似度
#         similarity = self.cosine_similarity(aggregated_features, co_embeddings.unsqueeze(1))
#         return similarity



# class PatientFeatureLayer(nn.Module):
#     def __init__(self, feature_input_dim, graph_size):
#         super().__init__()
#         # 特征自注意力机制
#         self.feature_atten = nn.MultiheadAttention(
#             embed_dim=feature_input_dim,
#             num_heads=4
#         )
#         # 特征值投影
#         self.value_proj = nn.Linear(1, graph_size)
#         # 特征重要性门控
#         self.gate = nn.Sequential(
#             nn.Linear(feature_input_dim, feature_input_dim),
#             nn.Sigmoid()
#         )
#
#
#     def forward(self, data_tensor, co_embeddings):
#         # 维度适配
#         data = data_tensor.unsqueeze(1)
#         # 特征自注意力
#         attn_output, _ = self.feature_atten(data, data, data)
#         attn_output = attn_output.squeeze(1) #(seq_len, 76)
#
#         # 时间维度聚合（保留特征）
#         global_feat = attn_output.mean(dim=0)
#
#         # 特征值投影（每个特征单独编码）
#         feat_values = global_feat.unsqueeze(-1)
#         project_feat = self.value_proj(feat_values)
#
#         # 疾病-特征相关性计算
#         corr_scores = torch.matmul(
#             co_embeddings,
#             project_feat.T
#         )
#
#         # # 动态特征门控
#         gate_weights = self.gate(global_feat)  # (embed_size,)
#         similarity = corr_scores * gate_weights.unsqueeze(0)
#         return similarity

# # 消融
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
        global_feat = data_tensor.mean(dim=0)

        # 特征值投影（每个特征单独编码）
        feat_values = global_feat.unsqueeze(-1)
        project_feat = self.value_proj(feat_values)


        # 疾病-特征相关性计算
        # 点积运算
        # corr_scores = torch.matmul(
        #     co_embeddings,
        #     project_feat.T
        # )

        # 余弦相似度
        corr_scores = self.cosine_sim(
            co_embeddings.unsqueeze(1),
            project_feat.unsqueeze(0)
        )

        # 动态特征门控
        gate_weights = self.gate(global_feat)  # (embed_size,)
        similarity = corr_scores * gate_weights.unsqueeze(0)
        return similarity


# class PatientFeatureLayer(nn.Module):
#     def __init__(self, hidden_dim, feature_input_dim, graph_size):
#         super().__init__()
#
#         self.cosine_similarity = nn.CosineSimilarity(dim=-1)
#         self.feature_input_dim = feature_input_dim
#
#         self.linear = nn.Linear(1, graph_size)
#         self.activation = nn.ReLU()
#         self.seq_processor = nn.Sequential(
#             nn.Linear(feature_input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, graph_size),
#             nn.Tanh()
#         )
#         # 图节点交互器
#         self.node_projector = nn.Linear(graph_size, feature_input_dim)
#
#         # 残差连接
#         self.residual = nn.Linear(feature_input_dim, feature_input_dim)
#
#     def forward(self, data_tensor, co_embeddings):
#         # 处理整个序列
#         seq_features = self.seq_processor(data_tensor)  # [seq_len, graph_hidden]
#
#         # 与图节点交互
#         node_weights = torch.einsum('cf,sf->cs',
#                                     co_embeddings,
#                                     seq_features)  # [code_num, seq_len]
#
#         # 注意力聚合
#         attn = torch.softmax(node_weights, dim=1)
#         aggregated = torch.einsum('cs,sf->cf',
#                                   attn,
#                                   seq_features)  # [code_num, graph_hidden]
#
#         # 特征重建
#         node_features = self.node_projector(aggregated)  # [code_num, feature_input_size]
#
#         # 残差连接原始特征（平均）
#         residual = self.residual(data_tensor.mean(dim=0))  # [feature_input_size]
#
#         return node_features + residual.unsqueeze(0)  # 广播相加 [code_num, feature_input_size]


# class PatientFeatureLayer(nn.Module):
#     def __init__(self, rnn_hidden_dim, feature_input_dim, graph_size):
#         super().__init__()
#         self.cosine_similarity = nn.CosineSimilarity(dim=-1)
#         self.feature_input_dim = feature_input_dim
#         self.linear = nn.Linear(1, graph_size)
#         self.activation = nn.ReLU()
#         self.mlp = nn.Sequential(
#             nn.Linear(feature_input_dim, rnn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(rnn_hidden_dim, feature_input_dim)
#         )
#
#
#     def forward(self, data_tensor, co_embeddings):
#         data = self.mlp(data_tensor)
#         feature_input = self.activation(self.linear(data.T.unsqueeze(-1)))
#         similarity = self.cosine_similarity(feature_input.mean(dim=1), co_embeddings.unsqueeze(1))
#         return similarity

# class PatientFeatureLayer(nn.Module):
#     def __init__(self, mlp_hidden_dim, feature_input_dim, graph_size):
#         super().__init__()
#         self.cosine_similarity = nn.CosineSimilarity(dim=-1)
#
#         # 定义MLP结构
#         self.mlp = nn.Sequential(
#             nn.Linear(feature_input_dim, mlp_hidden_dim),  # 特征维度提升
#             nn.ReLU(),  # 非线性激活
#             nn.Linear(mlp_hidden_dim, graph_size)  # 输出到图嵌入维度
#         )
#
#     def forward(self, data_tensor, co_embeddings):
#         # 输入数据形状: (seq_len, feature_input_dim)
#         processed = self.mlp(data_tensor)  # (seq_len, graph_size)
#         aggregated = processed.mean(dim=0)  # (graph_size,) 时间维度聚合
#
#         # 计算与所有医疗编码的余弦相似度
#         similarity = self.cosine_similarity(
#             aggregated.unsqueeze(0),  # 扩展维度 (1, graph_size)
#             co_embeddings  # (code_num, graph_size)
#         )  # 输出形状 (code_num,)
#
#         return similarity


# GRU版本
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

        # 处理共现诊断
        if len(m1_index) > 0:

            # # 对共现诊断也添加特征学习的结果
            # similarity_embeddings = self.feature_to_graph(similarity_matrix)
            # # 归一化处理
            # # similarity_embeddings_normalized = self.layer_norm_sim(similarity_embeddings)
            # # co_embeddings_normalized = self.layer_norm_co(co_embeddings)
            # # 线性融合
            # # fusion_embeddings = torch.cat([similarity_embeddings_normalized, co_embeddings_normalized], dim=-1)
            # # fo_embeddings = self.embedding_fusion(fusion_embeddings)
            # # 其他融合方式
            # fo_embeddings = self.embeddings_fusion_layer(similarity_embeddings, co_embeddings)
            # m1_embedding = fo_embeddings[m1_index]

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

# copy type 备份
# 与GRU版本的主要区别：
# 移除了对M1的额外特征处理（注释掉的代码部分）
# 在M2、M3处理中，取消了activation函数的应用
# 直接使用fusion_layer融合两种注意力输出，不经过额外的激活函数
# 结构更简洁，是GRU版本的简化版
# class TransitionLayer(nn.Module):
#     def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size, feature_size):
#         super().__init__()
#         self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)  # GRU
#         # 单头注意力机制，用于处理不同类型编码之间的交互。输入维度为 graph_size，注意力机制的中间层维度为 t_attention_size，输出维度为 t_output_size。
#         self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
#         self.cross_modal_attention = CrossModalAttention(graph_size, graph_size, t_output_size, t_attention_size)
#         self.feature_to_graph = nn.Linear(feature_size, graph_size)
#         self.fusion_layer = WeightedSumFusionLayer(hidden_size)
#         self.embeddings_fusion_layer = WeightedSumFusionLayer(graph_size)
#         self.embedding_fusion = nn.Linear(2 * graph_size, graph_size)
#         self.linear_fusion = nn.Linear(2 * hidden_size, hidden_size)
#         self.layer_norm_sim = nn.LayerNorm(graph_size)
#         self.layer_norm_co = nn.LayerNorm(graph_size)
#         self.layer_norm_hd = nn.LayerNorm(hidden_size)
#         self.layer_norm_hz = nn.LayerNorm(hidden_size)
#
#         self.activation = nn.Tanh()
#
#         self.code_num = code_num
#         self.hidden_size = hidden_size
#
#     def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, similarity_matrix,
#                 hidden_state=None):
#         # 分割索引`
#         m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
#         m1_index = torch.where(m1 > 0)[0]  # m1中大于0，表示共现编码的索引
#         m2_index = torch.where(m2 > 0)[0]  # m2中大于0，表示邻居编码的索引
#         m3_index = torch.where(m3 > 0)[0]  # m3中大于0，表示无关编码的索引
#         # 初始化隐藏状态
#         h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
#         output_m1 = 0
#         output_m23 = 0
#
#         # 处理共现诊断
#         if len(m1_index) > 0:
#
#             # 原方法
#             m1_embedding = co_embeddings[m1_index]
#             h = hidden_state[m1_index] if hidden_state is not None else None  # hidden_state不为None则赋值，即非第一时间步
#             h_m1 = self.gru(m1_embedding, h)
#
#             h_new[m1_index] = h_m1
#             output_m1, _ = torch.max(h_m1, dim=-2)  # 最大池化
#
#         # 处理新出现的邻居m2和无关诊断m3
#         if t > 0 and len(m2_index) + len(m3_index) > 0:
#             q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])  # q
#             v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])  # v
#             h_d = self.single_head_attention(q, q, v)
#
#             similarity_embeddings = self.feature_to_graph(similarity_matrix)
#             similarity_embeddings_normalized = self.layer_norm_sim(similarity_embeddings)
#             co_embeddings_normalized = self.layer_norm_co(co_embeddings)
#
#             # 其他融合方法
#             fo_embeddings = self.embeddings_fusion_layer(similarity_embeddings_normalized, co_embeddings_normalized)
#
#
#             # 结合特征学习
#             q_z = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
#             k_z = torch.vstack([fo_embeddings[m2_index], fo_embeddings[m3_index]])
#             v_z = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
#             h_z = self.cross_modal_attention(q_z, k_z, v_z)
#
#             h_m23 = self.fusion_layer(h_d, h_z)
#
#             # h_m23的前半部分分给m2,后半分给m3_index
#             h_new[m2_index] = h_m23[:len(m2_index)]
#             h_new[m3_index] = h_m23[len(m2_index):]
#
#             output_m23, _ = torch.max(h_m23, dim=-2)  # 计算h_m23最大值
#
#         if len(m1_index) == 0:
#             output = output_m23
#         elif len(m2_index) + len(m3_index) == 0:
#             output = output_m1
#         else:
#             output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
#         return output, h_new

# LSTM版本
# class TransitionLayer(nn.Module):
#     def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size, feature_size):
#         super().__init__()
#         self.lstm = nn.LSTMCell(input_size=graph_size, hidden_size=hidden_size)
#         # 单头注意力机制，用于处理不同类型编码之间的交互。输入维度为 graph_size，注意力机制的中间层维度为 t_attention_size，输出维度为 t_output_size。
#         self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
#         self.cross_modal_attention = CrossModalAttention(graph_size, graph_size, t_output_size, t_attention_size)
#         self.feature_to_graph = nn.Linear(feature_size, graph_size)
#         self.fusion_layer = WeightedSumFusionLayer(hidden_size)
#         self.embeddings_fusion_layer = WeightedSumFusionLayer(graph_size)
#         self.embedding_fusion = nn.Linear(2 * graph_size, graph_size)
#         self.linear_fusion = nn.Linear(2 * hidden_size, hidden_size)
#         self.layer_norm_sim = nn.LayerNorm(graph_size)
#         self.layer_norm_co = nn.LayerNorm(graph_size)
#         self.layer_norm_hd = nn.LayerNorm(hidden_size)
#         self.layer_norm_hz = nn.LayerNorm(hidden_size)
#
#         self.activation = nn.Tanh()
#
#         self.code_num = code_num
#         self.hidden_size = hidden_size
#
#     def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, similarity_matrix,
#                 hidden_state=None, c_state=None):
#         # 分割索引`
#         m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
#         m1_index = torch.where(m1 > 0)[0]  # m1中大于0，表示共现编码的索引
#         m2_index = torch.where(m2 > 0)[0]  # m2中大于0，表示邻居编码的索引
#         m3_index = torch.where(m3 > 0)[0]  # m3中大于0，表示无关编码的索引
#         # 初始化隐藏状态
#         h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
#         # 初始化细胞状态
#         c_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
#         output_m1 = 0
#         output_m23 = 0
#
#         # 处理共现诊断
#         if len(m1_index) > 0:
#             m1_embedding = co_embeddings[m1_index]
#             batch_size = m1_embedding.size(0)
#             if hidden_state is not None:
#                 h = hidden_state[m1_index]
#             else:
#                 h = torch.zeros(batch_size, self.hidden_size, dtype=co_embeddings.dtype).to(co_embeddings.device)
#             if c_state is not None:
#                 c = c_state[m1_index]
#             else:
#                 c = torch.zeros(batch_size, self.hidden_size, dtype=co_embeddings.dtype).to(co_embeddings.device)
#
#             h_m1, c_m1 = self.lstm(m1_embedding, (h, c))
#
#             h_new[m1_index] = h_m1
#             c_new[m1_index] = c_m1
#             output_m1, _ = torch.max(h_m1, dim=-2)  # 最大池化
#
#         # 处理新出现的邻居m2和无关诊断m3
#         if t > 0 and len(m2_index) + len(m3_index) > 0:
#             q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])  # q
#             v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])  # v
#             # h_m23 = self.activation(self.single_head_attention(q, q, v))
#             # c_m23 = self.activation(self.single_head_attention(q, q, v))
#
#             h_d = self.single_head_attention(q, q, v)
#
#             similarity_embeddings = self.feature_to_graph(similarity_matrix)
#             similarity_embeddings_normalized = self.layer_norm_sim(similarity_embeddings)
#             co_embeddings_normalized = self.layer_norm_co(co_embeddings)
#
#             # 其他融合方法
#             fo_embeddings = self.embeddings_fusion_layer(similarity_embeddings_normalized, co_embeddings_normalized)
#
#             # 结合特征学习
#             q_z = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
#             k_z = torch.vstack([fo_embeddings[m2_index], fo_embeddings[m3_index]])
#             v_z = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
#             h_z = self.cross_modal_attention(q_z, k_z, v_z)
#
#             # h_m23 = self.activation(self.fusion_layer(h_d, h_z))
#
#             h_m23 = self.activation(h_d)
#             c_m23 = self.activation(h_z)
#
#             # h_m23的前半部分分给m2,后半分给m3_index
#             h_new[m2_index] = h_m23[:len(m2_index)]
#             h_new[m3_index] = h_m23[len(m2_index):]
#
#             c_new[m2_index] = c_m23[:len(m2_index)]
#             c_new[m3_index] = c_m23[len(m2_index):]
#
#             output_m23, _ = torch.max(h_m23, dim=-2)  # 计算h_m23最大值
#
#         if len(m1_index) == 0:
#             output = output_m23
#         elif len(m2_index) + len(m3_index) == 0:
#             output = output_m1
#         else:
#             output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
#         return output, h_new, c_new


# 移除了所有与特征学习相关的组件
# 移除了cross_modal_attention、feature_to_graph、融合层等
# 前向传播移除了similarity_matrix参数
# 处理M2、M3时只使用简单的单头注意力
# 结构最简洁，只保留了GRU和基础注意力机制
# class TransitionLayer(nn.Module):
#     def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size, feature_size):
#         super().__init__()
#         self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)  # GRU
#         # 单头注意力机制，用于处理不同类型编码之间的交互。输入维度为 graph_size，注意力机制的中间层维度为 t_attention_size，输出维度为 t_output_size。
#         self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
#         self.cross_modal_attention = CrossModalAttention(graph_size, graph_size, t_output_size, t_attention_size)
#         self.feature_to_graph = nn.Linear(feature_size, graph_size)
#         # self.fusion_layer = FusionLayer(hidden_size)
#         self.fusion_layer = WeightedSumFusionLayer(hidden_size)
#         self.embeddings_fusion_layer = WeightedSumFusionLayer(graph_size)
#         self.embedding_fusion = nn.Linear(2 * graph_size, graph_size)
#         self.linear_fusion = nn.Linear(2 * hidden_size, hidden_size)
#         self.layer_norm_sim = nn.LayerNorm(graph_size)
#         self.layer_norm_co = nn.LayerNorm(graph_size)
#         self.layer_norm_hd = nn.LayerNorm(hidden_size)
#         self.layer_norm_hz = nn.LayerNorm(hidden_size)
#
#         self.activation = nn.Tanh()
#
#         self.code_num = code_num
#         self.hidden_size = hidden_size
#
#     def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings,
#                 hidden_state=None):
#         # 分割索引`
#         m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
#         m1_index = torch.where(m1 > 0)[0]  # m1中大于0，表示共现编码的索引
#         m2_index = torch.where(m2 > 0)[0]  # m2中大于0，表示邻居编码的索引
#         m3_index = torch.where(m3 > 0)[0]  # m3中大于0，表示无关编码的索引
#         # 初始化隐藏状态
#         h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
#         output_m1 = 0
#         output_m23 = 0
#
#         # 处理共现诊断
#         if len(m1_index) > 0:
#             # 原方法
#             m1_embedding = co_embeddings[m1_index]
#             h = hidden_state[m1_index] if hidden_state is not None else None  # hidden_state不为None则赋值，即非第一时间步
#             h_m1 = self.gru(m1_embedding, h)
#
#             h_new[m1_index] = h_m1
#             output_m1, _ = torch.max(h_m1, dim=-2)  # 最大池化
#
#         # 处理新出现的邻居m2和无关诊断m3
#         if t > 0 and len(m2_index) + len(m3_index) > 0:
#             q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])  # q
#             v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])  # v
#             h_m23 = self.activation(self.single_head_attention(q, q, v))
#
#             # h_m23的前半部分分给m2,后半分给m3_index
#             h_new[m2_index] = h_m23[:len(m2_index)]
#             h_new[m3_index] = h_m23[len(m2_index):]
#
#             output_m23, _ = torch.max(h_m23, dim=-2)  # 计算h_m23最大值
#
#         if len(m1_index) == 0:
#             output = output_m23
#         elif len(m2_index) + len(m3_index) == 0:
#             output = output_m1
#         else:
#             output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
#         return output, h_new
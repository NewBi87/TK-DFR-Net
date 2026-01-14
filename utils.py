import os
import pickle
import torch
import numpy as np
import random
import re

from preprocess import load_sparse
#多个辅助函数和类，它们在数据加载、学习率调度以及时间格式化等方面提供了重要的支持。
# 加载邻接矩阵adj,
# 原加载类
# def load_adj(path, device=torch.device('cpu')):
#     filename = os.path.join(path, 'code_adj.npz')
#     adj = torch.from_numpy(load_sparse(filename)).to(device=device, dtype=torch.float32)
#     return adj

# def load_adj(path, device = torch.device('cpu'), use_ontology=True, alpha=0.3):
#     """
#     修改后的图加载函数：融合统计图与本体图
#     """
#     # 加载原始统计图
#     filename = os.path.join(path, 'code_adj.npz')
#
#     # 调用原有的读取函数，返回的是numpy array
#     adj_stat_np = load_sparse(filename)
#
#     # 转为Tensor
#     adj_stat = torch.from_numpy(adj_stat_np).to(device=device, dtype=torch.float32)
#
#     # 2.加载并融合医学本体图
#     if use_ontology:
#         filename_ont = os.path.join(path, 'adj_ontology.npz')
#         if os.path.exists(filename_ont):
#             print(f"[info] Loading and fusing ontology graph from {filename_ont}...")
#
#             data_ont = np.load(filename_ont)
#             row = data_ont['row']
#             col = data_ont['col']
#             val = data_ont['data']
#             shape = data_ont['shape']
#
#             adj_ont_np = np.zeros(shape, dtype=np.float32)
#             adj_ont_np[row, col] = val
#
#             adj_ont = torch.from_numpy(adj_ont_np).to(device=device, dtype=torch.float32)
#
#             adj_stat = adj_stat + torch.eye(adj_stat.shape[0]).to(device)
#             adj_stat[adj_stat > 1] = 1
#
#             # 加权融合
#             adj = (1 - alpha) * adj_stat + alpha * adj_ont
#
#             ## 归一化
#             # 针对有向图/非对称矩阵，我们使用 D^-1 * A (行归一化)
#             # 这样每一行的和为1，代表从该节点流出的总概率/权重
#             rowsum = adj.sum(1)
#             r_inv = torch.pow(rowsum, -1).flatten()
#             r_inv[torch.isinf(r_inv)] = 0.
#             d_mat_inv = torch.diag(r_inv)
#
#             adj = torch.mm(d_mat_inv, adj)
#             return adj
#         else:
#             print("[Warning] Ontology graph not found, using original graph only.")
#     # 3. Fallback: 原始逻辑
#     adj_stat = adj_stat + torch.eye(adj_stat.shape[0]).to(device)
#     adj_stat[adj_stat > 1] = 1
#
#     # 对原始图也采用相同的行归一化，保持一致性
#     rowsum = adj_stat.sum(1)
#     r_inv = torch.pow(rowsum, -1).flatten()
#     r_inv[torch.isinf(r_inv)] = 0.
#     d_mat_inv = torch.diag(r_inv)
#     adj_stat = torch.mm(d_mat_inv, adj_stat)
#
#     return adj_stat

def load_adj(path, device=torch.device('cpu'), use_ontology=True, alpha=0.2):
    """
    修改后的图加载函数：融合统计图与本体图
    【融合策略】：叠加增强模式 (Add-on)，保留原图完整信息
    """
    # -------------------------------------------------------
    # 1. 加载原始统计图 (Base Graph)
    # -------------------------------------------------------
    filename_stat = os.path.join(path, 'code_adj.npz')
    adj_stat_np = load_sparse(filename_stat)
    adj_stat = torch.from_numpy(adj_stat_np).to(device=device, dtype=torch.float32)

    # 预处理：给统计图加上自循环 (Self-loop)
    # 无论是否融合，原图都应该有自循环
    adj_stat = adj_stat + torch.eye(adj_stat.shape[0]).to(device)
    adj_stat[adj_stat > 1] = 1

    # -------------------------------------------------------
    # 2. 加载并融合医学本体图 (如果开启)
    # -------------------------------------------------------
    if use_ontology:
        filename_ont = os.path.join(path, 'adj_ontology.npz')
        if os.path.exists(filename_ont):
            print(f"[Info] Loading and fusing ontology graph from {filename_ont}...")

            # 读取本体图
            data_ont = np.load(filename_ont)
            row = data_ont['row']
            col = data_ont['col']
            val = data_ont['data']
            shape = data_ont['shape']

            adj_ont_np = np.zeros(shape, dtype=np.float32)
            adj_ont_np[row, col] = val
            adj_ont = torch.from_numpy(adj_ont_np).to(device=device, dtype=torch.float32)

            # --- 【核心修改】叠加增强模式 ---
            # 旧逻辑 (Trade-off): adj = (1 - alpha) * adj_stat + alpha * adj_ont
            # 缺点：alpha=0.3时，原图权重只剩0.7，信息丢失严重

            # 新逻辑 (Add-on): adj = adj_stat + alpha * adj_ont
            # 优点：原图保持权重1.0，本体图作为额外 shortcut (权重0.05) 补充进来
            adj = adj_stat + alpha * adj_ont

            # --- 归一化 (Row-Normalize) ---
            # 针对有向图/非对称矩阵，使用 D^-1 * A
            rowsum = adj.sum(1)
            r_inv = torch.pow(rowsum, -1).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            d_mat_inv = torch.diag(r_inv)

            adj = torch.mm(d_mat_inv, adj)

            return adj
        else:
            print("[Warning] Ontology graph not found, using original graph only.")

    # -------------------------------------------------------
    # 3. Fallback: 仅使用统计图 (未开启融合 或 文件不存在)
    # -------------------------------------------------------
    # 归一化
    rowsum = adj_stat.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    d_mat_inv = torch.diag(r_inv)
    adj_stat = torch.mm(d_mat_inv, adj_stat)

    return adj_stat
# 自定义数据类，负责加载和预处理 EHR 数据，并提供迭代器接口以供训练和评估使用。
'''
添加功能：索引：保持 train_pids 和 train_code_x 之间的索引关系
在 EHRDataset 类中添加一个 pids 属性，并在 _load 方法中加载 pids.pkl 文件中的 train_pids 列表。然后在 __getitem__ 中返回相应的 pid，以便在需要时可以追踪到原始的患者 ID。
'''
class EHRDataset:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cpu')):
        super().__init__()
        self.path = data_path
        self.code_x, self.visit_lens, self.y, self.divided, self.neighbors, self.pids = self._load(label)

        self._size = self.code_x.shape[0]
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    # 从磁盘加载数据文件，包括诊断代码、就诊次数、标签、分割索引和邻居信息。
    def _load(self, label):
        # 判断当前数据集
        dataset_type = os.path.basename(os.path.normpath(self.path))
        if dataset_type not in['train','valid','test']:
            raise ValueError("Invalid dataset type. Expected 'train', 'valid', or 'test'.")
        # 获取父目录
        dataset_parent_path = os.path.dirname(os.path.dirname(self.path))

        # 构建 pids.pkl 的相对路径
        encoded_path = os.path.join(dataset_parent_path, 'encoded', 'pids.pkl')

        # 加载 pids.pkl 文件
        with open(encoded_path, 'rb') as f:
            pids_dict = pickle.load(f)

        # 根据 dataset_type 选择对应的 pids
        if dataset_type == 'train':
            pids = pids_dict['train_pids']
        elif dataset_type == 'valid':
            pids = pids_dict['valid_pids']
        elif dataset_type == 'test':
            pids = pids_dict['test_pids']

        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        if label == 'm':
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h':
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        divided = load_sparse(os.path.join(self.path, 'divided.npz'))
        neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'))
        return code_x, visit_lens, y, divided, neighbors, pids

    # 在每个epoch结束时打乱数据索引
    # pids 和 code_x 之间的对应关系不会被破坏，因为它们都是根据同一个 self.idx 来选择的。
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    # 返回数据集大小
    def size(self):
        return self._size

    # 返回标签
    def label(self):
        return self.y

    # 返回数据集的批次数量
    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    # 根据索引返回一个批次的数据，并将数据移动到指定设备。
    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        divided = torch.from_numpy(self.divided[slices]).to(device)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        pids = self.pids[slices]  # 返回对应的 pids
        return code_x, visit_lens, divided, y, neighbors, pids

class Reader:
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)
# 时间序列读取类
class TimeSeriesReader(Reader):
    def __init__(self, dataset_dir, listfile=None, device=torch.device('cpu')):
        super().__init__(dataset_dir, listfile)  # 使用 super() 调用父类构造函数
        self._data = [line.strip() for line in self._data]  # 修改 _data 的格式 #改line.split(',')为line.strip()
        self.device = device

    # 读取单个时间序列文件的内容
    # 接受文件名作为参数，返回的是时间序列数据和列名
    def _read_timeseries(self, ts_filename):
        ret = []
        data = {}
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            ret = []
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
                #ret.append(np.array(mas, dtype=np.float32))  # 确保数据类型为 float32
        #return (np.stack(ret), header)
            data["X"] = [np.stack(ret)]
            data["header"] = header
            data["name"] = ts_filename
        return data

    # 根据索引从listfile.csv中读取一个完整的样本呢
    # 不仅返回时间序列数据，还包括了文件名等额外的元信息。
    def read_example(self, index):
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index]
        #(X, header) = self._read_timeseries(name)
        data = self._read_timeseries(name)

        # return {"X": X,
        #         "header": header,
        #         "name": name}
        return data

    def get_indices_by_pid(self, pid):
        # 定义正则表达式模式，用于匹配文件名
        pattern = re.compile(rf"^{pid}_episode\d+_timeseries\.csv$")

        indices = []
        for i, filename in enumerate(self._data):
            if pattern.match(filename):
                indices.append(i)

        return indices

    def get_device(self):
        return self.device


# 多步学习率调度器，它可以根据预定义的里程碑（milestones）调整学习率。
class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer  #优化器对象。
        self.epochs = epochs    #总训练轮数。
        self.init_lr = init_lr  #初始学习率。
        self.lrs = self._generate_lr(milestones, lrs)   #学习率调整的时间点。
        self.current_epoch = 0  #对应于每个里程碑的学习率。

    # 根据里程碑生成学习率序列。
    def _generate_lr(self, milestones, lrs):
        milestones = [1] + milestones + [self.epochs + 1]
        lrs = [self.init_lr] + lrs
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i], )) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    # 在每个 epoch 结束时更新优化器的学习率。
    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_epoch += 1

    #重置当前 epoch 计数器。
    def reset(self):
        self.current_epoch = 0

# 将秒数格式化为更易读的时间字符串，适用于显示训练时间或剩余时间。
def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str


# ============================================================
# Innovation Point B: Evidential Deep Learning (EDL) Loss
# ============================================================
import torch.nn.functional as F


class BinaryEDLLoss(torch.nn.Module):
    def __init__(self, annealing_step=10, device=torch.device('cpu')):
        super().__init__()
        self.annealing_step = annealing_step
        self.device = device
        self.epoch = 0

    def update_epoch(self, epoch):
        """
        需要在每个 epoch 开始时调用此函数更新 epoch 数
        用于计算 KL 散度的退火系数 (annealing coefficient)
        """
        self.epoch = epoch

    # def forward(self, logits, target):
    #     """
    #     logits: 模型直接输出 (batch_size, num_labels)，注意：不要经过 Sigmoid！
    #     target: 真实标签 (batch_size, num_labels)，0 或 1
    #     """
    #     # 1. 将 Logits 分解为正负证据 (Evidence)
    #     # softplus 保证证据是非负的
    #     evidence_pos = F.softplus(logits)
    #     evidence_neg = F.softplus(-logits)
    #
    #     # 2. 构建 Beta 分布参数 (alpha, beta)
    #     # alpha = 正证据 + 1
    #     # beta  = 负证据 + 1
    #     alpha = evidence_pos + 1
    #     beta = evidence_neg + 1
    #
    #     # S = 总证据强度 (Total Evidence)
    #     S = alpha + beta
    #
    #     # 3. 预测概率 (Expectation of Beta distribution)
    #     prob = alpha / S
    #
    #     # --- 计算损失项 ---
    #
    #     # A. 均方误差风险 (MSE Risk)
    #     # 目标是 1 时，希望 prob 接近 1；目标是 0 时，希望 prob 接近 0
    #     risk = (target - prob) ** 2
    #
    #     # B. 不确定性方差 (Variance Risk)
    #     # 当总证据 S 越大，方差越小，模型越确定
    #     variance = (prob * (1 - prob)) / (S + 1)
    #
    #     # C. KL 散度正则化 (KL Divergence Regularization)
    #     # 防止模型在“不知道”的时候强行给出一个错误的高置信度
    #     # 我们希望当没有证据时，分布接近均匀分布 (alpha=1, beta=1)
    #
    #     # 计算退火系数：随着训练进行，KL 正则化的权重逐渐增加
    #     # 防止训练初期模型因为正则化太强而学不到东西
    #     annealing_coef = min(1.0, self.epoch / self.annealing_step)
    #
    #     # 近似计算：仅对“误导性证据”进行惩罚
    #     # 如果标签是1，我们惩罚 negative evidence；如果标签是0，我们惩罚 positive evidence
    #     misleading_evidence = target * evidence_neg + (1 - target) * evidence_pos
    #     kl_penalty = 0.01 * misleading_evidence
    #
    #     # 总损失 = 风险 + 方差 + 退火系数 * KL惩罚
    #     loss = torch.mean(risk + variance + annealing_coef * kl_penalty)
    #
    #     return loss
    def forward(self, logits, target):
        """
        logits: 模型直接输出 (batch_size, num_labels)
        target: 真实标签 (batch_size, num_labels)
        """
        # 1. 获取证据 (Evidence)
        evidence_pos = F.softplus(logits)
        evidence_neg = F.softplus(-logits)

        # 2. 构建 Beta 分布参数
        alpha = evidence_pos + 1
        beta = evidence_neg + 1
        S = alpha + beta
        prob = alpha / S

        # --- 计算损失项 ---

        # A. 均方误差风险 (MSE Risk) - 【核心修改点】
        # MIMIC 数据极度不平衡，MSE 容易导致模型预测全 0。
        # 我们给正样本 (target=1) 增加权重，强迫模型关注少数类。
        # 经验值：10 到 20 倍是比较合适的
        pos_weight = 20.0

        # 如果是正样本，Loss 放大 20 倍；负样本保持 1 倍
        weight = target * pos_weight + (1 - target) * 1.0

        # 加权后的 MSE
        risk = weight * ((target - prob) ** 2)

        # B. 不确定性方差 (Variance Risk)
        variance = (prob * (1 - prob)) / (S + 1)

        # C. KL 散度正则化
        annealing_coef = min(1.0, self.epoch / self.annealing_step)
        misleading_evidence = target * evidence_neg + (1 - target) * evidence_pos

        # 保持 1e-4 的缩放，这已经是安全的了
        scaling_factor = 1e-4
        kl_penalty = scaling_factor * misleading_evidence

        # 总损失
        loss = torch.mean(risk + variance + annealing_coef * kl_penalty)

        return loss

# --- 预备代码：Plan B 对比学习 Loss ---
class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (batch_size, hidden_dim) - 必须归一化
        labels: (batch_size, num_codes) - Multi-hot
        """
        # 1. 特征归一化 & 相似度计算
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 2. 标签相似度 (Jaccard)
        labels = labels.float()
        intersection = torch.matmul(labels, labels.T)
        union = labels.sum(1, keepdim=True) + labels.sum(1, keepdim=True).T - intersection
        label_sim = intersection / (union + 1e-8)

        # 3. Mask (去掉自身) & Log-Sum-Exp
        mask = torch.eye(labels.shape[0], device=labels.device).bool()
        label_sim.masked_fill_(mask, 0)

        exp_sim = torch.exp(sim_matrix) * (1 - mask.float())
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # 4. 最终 Loss (加权平均)
        mean_log_prob_pos = (label_sim * log_prob).sum(1) / (label_sim.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()
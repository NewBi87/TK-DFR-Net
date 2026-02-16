import os
import pickle
import torch
import numpy as np
import random
import re

from preprocess import load_sparse
#多个辅助函数和类，它们在数据加载、学习率调度以及时间格式化等方面提供了重要的支持。
# 加载邻接矩阵adj,
# 文件: utils.py

def load_adj(path, device=torch.device('cpu'), use_ontology=True, alpha=0.05):
    """
    [修正后] 双图模式：返回 (adj_stat, adj_ont) 两个独立的图，交给模型去融合
    """
    # 1. 加载统计图 (Base Graph)
    filename_stat = os.path.join(path, 'code_adj.npz')
    adj_stat_np = load_sparse(filename_stat)
    adj_stat = torch.from_numpy(adj_stat_np).to(device=device, dtype=torch.float32)

    # 归一化统计图 (Base Normalization)
    adj_stat = adj_stat + torch.eye(adj_stat.shape[0]).to(device)
    adj_stat[adj_stat > 1] = 1
    rowsum = adj_stat.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    d_mat_inv = torch.diag(r_inv)
    adj_stat = torch.mm(d_mat_inv, adj_stat)

    # 2. 加载本体图 (Ontology Graph)
    # 默认兜底：如果没有本体图，第二个图也用统计图
    adj_ont = adj_stat

    if use_ontology:
        filename_ont = os.path.join(path, 'adj_ontology.npz')
        if os.path.exists(filename_ont):
            print(f"[Info] Loading ontology graph for Dual-Graph Fusion: {filename_ont}")
            data_ont = np.load(filename_ont)
            # 重建矩阵
            adj_ont_np = np.zeros(data_ont['shape'], dtype=np.float32)
            adj_ont_np[data_ont['row'], data_ont['col']] = data_ont['data']
            adj_ont = torch.from_numpy(adj_ont_np).to(device=device, dtype=torch.float32)

            # 归一化本体图 (Ontology Normalization)
            adj_ont = adj_ont + torch.eye(adj_ont.shape[0]).to(device)
            adj_ont[adj_ont > 1] = 1
            rowsum_ont = adj_ont.sum(1)
            r_inv_ont = torch.pow(rowsum_ont, -1).flatten()
            r_inv_ont[torch.isinf(r_inv_ont)] = 0.
            d_mat_inv_ont = torch.diag(r_inv_ont)
            adj_ont = torch.mm(d_mat_inv_ont, adj_ont)
        else:
            print("[Warning] Ontology graph not found, Dual-Graph layer will use duplicate stat graphs.")

    # [关键] 返回元组 (Tuple)，而不是相加后的单个矩阵
    return adj_stat, adj_ont
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

def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


# ============================================================
# Innovation Point B: Evidential Deep Learning (EDL) Loss
# ============================================================
import torch.nn.functional as F


# class BinaryEDLLoss(torch.nn.Module):
#     def __init__(self, annealing_step=10, kl_weight=1e-4, device=torch.device('cpu')):
#         """
#         Args:
#             annealing_step: 退火步数
#             kl_weight: KL 散度的最大权重系数。[关键修改] 建议设为 1e-4 或 1e-5
#             device: 设备
#         """
#         super().__init__()
#         self.annealing_step = annealing_step
#         self.kl_weight = kl_weight  # [新增] 保存权重参数
#         self.device = device
#         self.epoch = 0
#
#     def update_epoch(self, epoch):
#         self.epoch = epoch
#
#     def forward(self, logits, target):
#         """
#         [高性能版] 使用 Digamma Loss + Logits Clamp + 自适应 KL 权重
#         """
#         # [修改] 限制 logits 范围，防止 evidence 过大导致 digamma 计算溢出
#         # 范围 [-10, 10] 足以覆盖 sigmoid 0.000045 到 0.999955 的区间，非常安全
#         logits = torch.clamp(logits, min=-10, max=10)
#
#         # 1. 获取证据 (Evidence)
#         evidence_pos = torch.nn.functional.softplus(logits)
#         evidence_neg = torch.nn.functional.softplus(-logits)
#
#         alpha = evidence_pos + 1
#         beta = evidence_neg + 1
#         S = alpha + beta
#
#         # 2. Digamma Loss (主任务损失)
#         # 数学原理：最小化 E[log(p)]，即最大化似然
#         digamma_S = torch.digamma(S)
#         digamma_alpha = torch.digamma(alpha)
#         digamma_beta = torch.digamma(beta)
#
#         # 当 target=1 时，优化 alpha 部分；当 target=0 时，优化 beta 部分
#         risk = target * (digamma_S - digamma_alpha) + (1 - target) * (digamma_S - digamma_beta)
#
#         # 3. KL 散度正则化 (防止模型过度自信)
#         # 计算当前 epoch 的退火系数 (0.0 -> 1.0)
#         annealing_coef = min(1.0, self.epoch / self.annealing_step)
#
#         # 计算 Beta 分布的 KL 散度 (Uniform Prior Beta(1,1))
#         # 近似计算公式
#         kl_alpha = (alpha - 1) * (1 - target)
#         kl_beta = (beta - 1) * target
#
#         # [核心修改] 使用 self.kl_weight
#         # 原来的 0.001 在 4000 个标签累加下依然太大，导致模型不敢预测。
#         # 这里改用传入的参数 (默认 1e-4)，相当于给 KL 惩罚再除以 10 倍。
#         kl_penalty = annealing_coef * self.kl_weight * (kl_alpha + kl_beta)
#
#         # 4. 总损失
#         # mean 会对 batch_size * num_labels 取平均
#         loss = torch.mean(risk + kl_penalty)
#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryEDLLoss(nn.Module):
    def __init__(self, annealing_step=10, kl_weight=1e-4, aux_weight=1.0, device=torch.device('cpu')):
        """
        aux_weight: 辅助损失的权重。对于 Focal Loss，建议设大一点 (如 0.5 ~ 1.0)，因为它数值通常比 BCE 小。
        """
        super().__init__()
        self.annealing_step = annealing_step
        self.kl_weight = kl_weight
        self.aux_weight = aux_weight
        self.device = device
        self.epoch = 0

    def update_epoch(self, epoch):
        self.epoch = epoch

    def focal_loss(self, logits, target, gamma=2.0, alpha=0.25):
        """
        二分类 Focal Loss 实现
        gamma: 聚焦参数，越大越关注难分样本 (Default: 2.0)
        alpha: 正负样本平衡参数 (Default: 0.25)
        """
        probs = torch.sigmoid(logits)

        # 计算 focal term
        # pt: 模型对正确类别的预测概率
        pt = torch.where(target == 1, probs, 1 - probs)

        # alpha_t: 平衡因子
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)

        # Loss = - alpha_t * (1 - pt)^gamma * log(pt)
        loss = - alpha_t * (1 - pt) ** gamma * torch.log(pt + 1e-8)

        return torch.mean(loss)

    def forward(self, logits, target):
        """
        [混合版 V2] MSE (EDL) + Focal Loss (Ranking)
        """
        # --- A. EDL 部分 (MSE) ---
        evidence = torch.relu(logits)
        alpha = evidence + 1
        S = alpha + 1
        prob = alpha / S

        # MSE Loss
        loss_mse = torch.mean((target - prob) ** 2 + prob * (1 - prob) / (S + 1))

        # KL 散度
        annealing_coef = min(1.0, self.epoch / self.annealing_step)
        kl_penalty = self.kl_weight * annealing_coef * evidence * (1 - target)

        loss_edl = loss_mse + torch.mean(kl_penalty)

        # --- B. Ranking 部分 (Focal Loss) ---
        # 专门针对 "硬负样本" 进行惩罚，优化 Top-K 排序
        loss_focal = self.focal_loss(logits, target)

        # --- C. 总损失 ---
        loss = loss_edl + self.aux_weight * loss_focal

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


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        # [修复] 兼容 Tensor 和 Numpy，增加 CPU 转换
        if isinstance(x, torch.Tensor):
            token = x[l - 1].detach().cpu().numpy()
        else:
            token = x[l - 1]
        result[i] = token
    return result
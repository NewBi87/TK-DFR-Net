import os
import pickle
import numpy as np
from icd9cms.icd9 import search

# 你的路径是标准的，直接用这个即可
base_path = 'data/mimic3'


def save_sparse(path, x):
    # 手动实现保存功能，避免依赖项目其他模块
    idx = np.where(x != 0)
    val = x[idx]
    np.savez(path, row=idx[0], col=idx[1], data=val, shape=x.shape)


# def generate_ontology_adj():
#     encoded_path = os.path.join(base_path, 'encoded', 'code_map.pkl')
#     output_path = os.path.join(base_path, 'standard', 'adj_ontology')
#
#     if not os.path.exists(encoded_path):
#         print(f"错误: 找不到文件 {encoded_path}")
#         return
#
#     print(f"Loading code map from {encoded_path}...")
#     with open(encoded_path, 'rb') as f:
#         # 兼容性加载
#         try:
#             code_map = pickle.load(f)
#         except UnicodeDecodeError:
#             code_map = pickle.load(f, encoding='latin1')
#
#     code_num = len(code_map)
#     print(f"Found {code_num} codes. Building ontology graph...")
#
#     adj = np.eye(code_num, dtype=np.int8)
#     code_list = list(code_map.keys())
#
#     # 建立 ICD9 查询缓存
#     parents = {}
#     for idx, code_str in enumerate(code_list):
#         # 移除小数点以匹配库格式 (例如 414.01 -> 41401)
#         clean_code = code_str.replace('.', '')
#         node = search(clean_code)
#         parents[idx] = node.parent if node else None
#
#     # 构建边 (兄弟节点互连)
#     count = 0
#     for i in range(code_num):
#         parent_i = parents[i]
#         if not parent_i: continue
#         for j in range(i + 1, code_num):
#             parent_j = parents[j]
#             if not parent_j: continue
#
#             if parent_i.code == parent_j.code:
#                 adj[i, j] = 1
#                 adj[j, i] = 1
#                 count += 1
#
#     print(f"Added {count} edges based on ontology.")
#     save_sparse(output_path, adj)
#     print(f"Success! Saved ontology adjacency matrix to {output_path}.npz")

# 文件: preprocess/generate_ontology.py

def generate_ontology_adj():
    encoded_path = os.path.join(base_path, 'encoded', 'code_map.pkl')
    output_path = os.path.join(base_path, 'standard', 'adj_ontology')

    if not os.path.exists(encoded_path):
        print(f"错误: 找不到文件 {encoded_path}")
        return

    print(f"Loading code map from {encoded_path}...")
    with open(encoded_path, 'rb') as f:
        try:
            code_map = pickle.load(f)
        except UnicodeDecodeError:
            code_map = pickle.load(f, encoding='latin1')

    code_num = len(code_map)
    print(f"Found {code_num} codes. Building ontology graph...")

    # 初始化邻接矩阵
    adj = np.eye(code_num, dtype=np.int8)
    code_list = list(code_map.keys())

    # 1. 建立 ICD9 节点缓存 (加速查询)
    icd9_nodes = {}
    for idx, code_str in enumerate(code_list):
        # 去除小数点以匹配库格式 (如 428.0 -> 4280)
        clean_code = code_str.replace('.', '')
        node = search(clean_code)
        if node:
            icd9_nodes[code_str] = node

    # 建立 code 字符串到索引的反向映射
    code_to_idx = {code: idx for idx, code in enumerate(code_list)}

    count_sibling = 0
    count_parent = 0

    # 2. 构建边 (包含兄弟关系 和 父子关系)
    for i in range(code_num):
        code_str_i = code_list[i]
        if code_str_i not in icd9_nodes: continue

        node_i = icd9_nodes[code_str_i]
        parent_i = node_i.parent  # 获取父节点对象

        # 如果没有父节点，跳过
        if not parent_i: continue

        # --- 修改点 A: 兄弟节点连接 (原逻辑保留) ---
        for j in range(i + 1, code_num):
            code_str_j = code_list[j]
            if code_str_j not in icd9_nodes: continue
            node_j = icd9_nodes[code_str_j]
            parent_j = node_j.parent

            # 如果两个节点的父节点代码相同，则是兄弟
            if parent_j and parent_i.code == parent_j.code:
                adj[i, j] = 1
                adj[j, i] = 1
                count_sibling += 1

        # --- 修改点 B: 父子节点连接 (新增逻辑) ---
        # 尝试匹配父节点是否在我们的代码列表中
        # 父节点可能有两种格式：带点 (428.0) 或 不带点 (4280)
        potential_parents = []
        if parent_i.code:
            potential_parents.append(parent_i.code)  # 原始格式
            if len(parent_i.code) > 3:
                # 尝试加点的格式 (如 4280 -> 428.0)
                potential_parents.append(parent_i.code[:3] + '.' + parent_i.code[3:])

        for p_code in potential_parents:
            if p_code in code_to_idx:
                p_idx = code_to_idx[p_code]
                # 建立双向连接 (i 是子, p_idx 是父)
                adj[i, p_idx] = 1
                adj[p_idx, i] = 1
                count_parent += 1

    print(f"Added {count_sibling} sibling edges and {count_parent} parent-child edges.")
    save_sparse(output_path, adj)
    print(f"Success! Saved ontology adjacency matrix to {output_path}.npz")

if __name__ == '__main__':
    generate_ontology_adj()
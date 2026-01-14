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


def generate_ontology_adj():
    encoded_path = os.path.join(base_path, 'encoded', 'code_map.pkl')
    output_path = os.path.join(base_path, 'standard', 'adj_ontology')

    if not os.path.exists(encoded_path):
        print(f"错误: 找不到文件 {encoded_path}")
        return

    print(f"Loading code map from {encoded_path}...")
    with open(encoded_path, 'rb') as f:
        # 兼容性加载
        try:
            code_map = pickle.load(f)
        except UnicodeDecodeError:
            code_map = pickle.load(f, encoding='latin1')

    code_num = len(code_map)
    print(f"Found {code_num} codes. Building ontology graph...")

    adj = np.eye(code_num, dtype=np.int8)
    code_list = list(code_map.keys())

    # 建立 ICD9 查询缓存
    parents = {}
    for idx, code_str in enumerate(code_list):
        # 移除小数点以匹配库格式 (例如 414.01 -> 41401)
        clean_code = code_str.replace('.', '')
        node = search(clean_code)
        parents[idx] = node.parent if node else None

    # 构建边 (兄弟节点互连)
    count = 0
    for i in range(code_num):
        parent_i = parents[i]
        if not parent_i: continue
        for j in range(i + 1, code_num):
            parent_j = parents[j]
            if not parent_j: continue

            if parent_i.code == parent_j.code:
                adj[i, j] = 1
                adj[j, i] = 1
                count += 1

    print(f"Added {count} edges based on ontology.")
    save_sparse(output_path, adj)
    print(f"Success! Saved ontology adjacency matrix to {output_path}.npz")


if __name__ == '__main__':
    generate_ontology_adj()
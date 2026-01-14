import os

import numpy as np
import h5py

def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)


def load_sparse(path):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat


def save_patient_data(base_path, pid, code_x, visit_lens, codes_y, hf_y, divided, neighbors):
    patient_dir = os.path.join(base_path, str(pid))
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    # 保存 code_x
    save_sparse(os.path.join(patient_dir, 'code_x'), code_x)
    # 保存 visit_lens
    np.savez(os.path.join(patient_dir, 'visit_lens'), lens=visit_lens)
    # 保存 codes_y
    save_sparse(os.path.join(patient_dir, 'codes_y'), codes_y)
    np.savez(os.path.join(base_path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(base_path, 'divided'), divided)
    save_sparse(os.path.join(base_path, 'neighbors'), neighbors)


def load_patient_data(base_path, pid):
    patient_dir = os.path.join(base_path, str(pid))
    # 加载 code_x
    code_x = load_sparse(os.path.join(patient_dir, 'code_x.npz'))
    # 加载 visit_lens
    visit_lens = np.load(os.path.join(patient_dir, 'visit_lens.npz'))['lens']
    # 加载 codes_y
    codes_y = load_sparse(os.path.join(patient_dir, 'codes_y.npz'))

    return code_x, visit_lens, codes_y

def save_data(path, code_x, visit_lens, codes_y, hf_y, divided, neighbors):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)


# 保存函数
def save_timeseries_data(data, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, f"{name}.hdf5")
    with h5py.File(file_path, 'w') as f:
        for key, value in data.items():
            f.create_dataset(str(key), data=value, compression="gzip", compression_opts=9)


# 读取函数
def load_timeseries_data(path, name):
    file_path = os.path.join(path, f"{name}.hdf5")
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data[int(key)] = f[key][:].astype(np.float32)
    return data

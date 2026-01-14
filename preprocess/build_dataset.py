import numpy as np

from preprocess.parse_csv import EHRParser


def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed)
    # 找到在code_map中至少有一个代码匹配的病人的id，并将这些病人添加到common_pids集合中
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():       #遍历每一个病人及其对应的住院记录
            for admission in admissions:                        #遍历当前病人的每一条住院记录
                codes = admission_codes[admission[EHRParser.adm_id_col]]    #从住院记录中获取与当前住院ID相关的所有代码
                if code in codes:           #检查当前code是否出现在该住院记录的代码列表中。
                    common_pids.add(pid)    #如果找到了匹配的代码，立即将病人的ID (pid) 添加到common_pids集合中，表示该病人的住院记录中包含至少一个匹配的代码。
                    break
            else:
                continue        # 一旦找到了一个匹配的病人，结束对该病人的循环，转向下一个code进行处理。
            break
    print('\r\t100%')
    #找出拥有最多住院记录的病人，并且将其加入到common_pids集合中，然后对剩余病人进行随机化处理
    #确保有最多住院记录的病人被优先处理
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids

#将病人的住院记录编码成模型训练所需的输入特征x和标签y，并且记录每个病人住院次数（除去最后一次）的长度lens
def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    pid_array = np.array(pids)  # 创建一个与 pids 对应的数组

    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens


def build_heart_failure_y(hf_prefix, codes_y, code_map):
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y

import pandas as pd
from preprocess.data_preprocessing  import pad_zeros
from tqdm import tqdm
import re

def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)

def calibrate_admission_codes(patient_admission, admission_codes):
    print(f"calibrating admission codes by patients again...")
    adm_id_set = set()

    # 收集所有有效的住院ID
    for admissions in patient_admission.values():
        for admission in admissions:
            adm_id_set.add(admission['adm_id'])

    #生成一个列表 del_adm_ids,其中包含所有在admission_codes中但不在adm_id_set中的住院ID
    del_adm_ids = [adm_id for adm_id in admission_codes if adm_id not in adm_id_set]

    # 从admission_codes中删除无效的住院ID
    for adm_id in del_adm_ids:
        del admission_codes[adm_id]
    print(f"Removed {len(del_adm_ids)} invalid admission IDs from admission_codes.")
    return admission_codes

def read_patient_data(reader, indices):
    data = {}
    for index in indices:
        ret = reader.read_example(index)
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

def preprocess_data(reader, discretizer, normalizer, pid_index):
    patient_data = {}
    for pid in tqdm(pid_index, desc="Processing patients", unit="patient"):
        indices = reader.get_indices_by_pid(pid)
        ret = read_patient_data(reader, indices)
        data = ret['X']
        data = [discretizer.transform(X)[0] for X in data]
        if (normalizer is not None):
            data = [normalizer.transform(X) for X in data]
        data_process = pad_zeros(data)
        patient_data[pid] = data_process
    return patient_data
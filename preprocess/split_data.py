import os
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd

def move_to_partition(root_path, patients, partition):
    if not os.path.exists(os.path.join(root_path, partition)):
        os.mkdir(os.path.join(root_path, partition))
    for patient in patients:
        src = os.path.join(root_path, patient)
        dest = os.path.join(root_path, partition, patient)
        shutil.move(src, dest)


def split_patient_timeseries(subjects_root_path, train_pids, valid_pids, test_pids):
    # 将NumPy数组转换为集合以加速查找
    train_set = set(map(str, train_pids))
    valid_set = set(map(str, valid_pids))
    test_set = set(map(str, test_pids))

    # 确保没有重复的患者ID
    assert len(train_set & valid_set) == 0, "Overlap between train and validation sets."
    assert len(train_set & test_set) == 0, "Overlap between train and test sets."
    assert len(valid_set & test_set) == 0, "Overlap between validation and test sets."

    # 获取所有主题文件夹，并过滤出名称为纯数字的文件夹
    folders = os.listdir(subjects_root_path)
    folders = list(filter(str.isdigit, folders))

    # 分别处理训练集、验证集和测试集
    for partition, pids in zip(['train', 'valid', 'test'], [train_set, valid_set, test_set]):
        patients = [x for x in folders if x in pids]
        move_to_partition(subjects_root_path, patients, partition)

def process_partition(root_path, output_path, partition):
    output_dir = os.path.join(output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    x_triples = []    #每个样本包含时间序列文件名、住院时长

    #获取病人列表
    patients = list(filter(str.isdigit, os.listdir(os.path.join(root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(root_path, partition, patient)

        # 检查患者文件夹是否存在
        if not os.path.exists(patient_folder):
            continue

        # 获取时序文件
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        if not patient_ts_files:
            print(f"No timeseries files found for patient {patient}.")
            continue

        for ts_filename in patient_ts_files:
            ts_filepath = os.path.join(patient_folder, ts_filename)

            # 检查文件是否存在
            if not os.path.exists(ts_filepath):
                print(f"Timeseries file {ts_filename} does not exist for patient {patient}.")
                continue

            try:
                with open(ts_filepath, 'r') as tsfile:
                    ts_lines = tsfile.readlines()
                    header = ts_lines[0].strip()
                    ts_lines = ts_lines[1:]

                    # 检查文件是否有有效数据行
                    if not ts_lines:
                        continue

                    # 保存处理后的时间序列数据
                    output_ts_filename = f"{patient}_{ts_filename}"
                    output_ts_filepath = os.path.join(output_dir, output_ts_filename)

                    with open(output_ts_filepath, 'w') as outfile:
                        outfile.write(header + '\n')
                        for line in ts_lines:
                            outfile.write(line)

                    # 添加样本到列表
                    x_triples.append(output_ts_filename)

            except Exception as e:
                print(f"Error processing timeseries file {ts_filename} for patient {patient}: {e}")


    print(f"Number of created samples: {len(x_triples)}")

    # 生成listfile.csv文件，仅包含第一列（时间序列文件名）
    listfile_header = "admit"
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for x in x_triples:
            listfile.write('{}\n'.format(x))

import os
import torch
import numpy as np
import random
import re

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
    def __init__(self, dataset_dir, listfile=None):
        super().__init__(dataset_dir, listfile)  # 使用 super() 调用父类构造函数
        self._data = [line.strip() for line in self._data]  # 修改 _data 的格式 #改line.split(',')为line.strip()

    # 读取单个时间序列文件的内容
    # 接受文件名作为参数，返回的是时间序列数据和列名
    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)


    # 根据索引从listfile.csv中读取一个完整的样本呢
    # 不仅返回时间序列数据，还包括了文件名等额外的元信息。
    def read_example(self, index):
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "header": header,
                "name": name}

    def get_indices_by_pid(self, pid):
        indices = []
        for i, filename in enumerate(self._data):
            if f"{pid}_episode" in filename:
                indices.append(i)
        return indices
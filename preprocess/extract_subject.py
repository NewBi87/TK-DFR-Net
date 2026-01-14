import numpy as np
import os
import pandas as pd

from .util import dataframe_from_csv


#读取病人的事件数据（events.csv），并进行初步清理，如删除缺失值、转换时间字段等。
def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events

# 过滤掉那些有转科记录的ICU住院记录，只保留没有转科的记录
def remove_icustays_with_transfers(stays):
    stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]

#读取病人的事件数据（events.csv），并进行初步清理，如删除缺失值、转换时间字段等。
def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events

# 从所有事件中提取与特定住院记录相关的事件，并可以根据入住事件和出院时间进一步筛选事件
def get_events_for_admit(events, admit_id, intime=None, outtime=None):
    idx = (events.HADM_ID == admit_id)
    # 如果提供了入住时间和出院时间，则进一步筛选出 CHARTTIME 在这两个时间之间的事件。
    if intime is not None and outtime is not None:
        time_filter = (events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime)
        idx = idx & time_filter  # 使用逻辑与运算符来确保两个条件都满足
    #过滤
    events_filtered = events[idx]
    #删除事件ID列
    if 'HADM_ID' in events_filtered:
        del events_filtered['HADM_ID']
    return events_filtered

#为事件数据添加一个新的列 HOURS，表示每个事件距离指定时间 dt 的小时数。可以选择是否删除原始的 CHARTTIME 列。
def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()  #为了避免对原始 events DataFrame 进行直接修改，我们首先创建一个副本。
    #计算每个事件的 CHARTTIME 与指定时间 dt 之间的差值。结果是一个 timedelta64 类型的 Series，表示时间差。
    events['HOURS'] = (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60  #转秒数后X60转换为小时
    if remove_charttime:
        del events['CHARTTIME']
    return events


#将事件数据转换为时间序列格式。
# 根据 CHARTTIME 和 VARIABLE 列将数据透视为宽表格式，其中每一列代表一个变量，每一行代表一个时间点。此外，它还会保留 ICUSTAY_ID 作为元数据。
def convert_events_to_timeseries(events, variable_column='VARIABLE', variables=[]):
    metadata = events[['CHARTTIME', 'HADM_ID']].sort_values(by=['CHARTTIME', 'HADM_ID'])\
                    .drop_duplicates(keep='first').set_index('CHARTTIME')   #按 CHARTTIME 和 ICUSTAY_ID 排序，确保时间顺序正确。
    # 从 events 中提取 CHARTTIME、VARIABLE 和 VALUE 三列，用于构建时间序列数据。
    timeseries = events[['CHARTTIME', variable_column, 'VALUE']]\
                    .sort_values(by=['CHARTTIME', variable_column, 'VALUE'], axis=0)\
                    .drop_duplicates(subset=['CHARTTIME', variable_column], keep='last')    #去除 CHARTTIME 和 VARIABLE 的重复组合，只保留最后一次出现的记录。这一步是为了确保每个时间点和每个变量的唯一性，并且保留最新的测量值。
    #!确保每个时间点和每个变量的唯一性，并且保留最新的测量值。是什么意思？仅保留最后的？
    #答：为了确保每个时间点（CHARTTIME）和每个变量（VARIABLE）的唯一性，并且保留最新的测量值。

    #透视数据并合并元数据
    timeseries = timeseries.pivot(index='CHARTTIME', columns=variable_column, values='VALUE')\
                    .merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    # 添加缺失的变量列
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

#从时间序列数据中提取某个变量的第一个有效值
def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
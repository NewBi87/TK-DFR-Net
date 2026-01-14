import numpy as np
import re

from pandas import DataFrame, Series

from .util import dataframe_from_csv

###############################
# Non-time series preprocessing
# 非时间序列数据预处理
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str): #简化复杂的种族/民族字符串
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}

# 组装周期性数据
def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.ICUSTAY_ID, 'Age': stays.AGE, 'Length of Stay': stays.LOS,
            'Mortality': stays.MORTALITY}
    data.update(transform_gender(stays.GENDER)) #更新性别
    data.update(transform_ethnicity(stays.ETHNICITY))   #种族
    data['Height'] = np.nan         #由于这些信息可能不是每次都可用，因此先将它们设置为NaN（Not a Number），表示缺失值。稍后这些字段会根据时间序列数据进行更新。
    data['Weight'] = np.nan
    data = DataFrame(data).set_index('Icustay') #转换为DataFranme，并设置Icustay索引
    #确保DataFrame的列按照期望的顺序排列，以便于后续处理和查看。
    data = data[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)

# 创建一个二进制矩阵，表示病人是否具有某个特定的诊断
diagnosis_labels = ['4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
                    '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
                    '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
                    '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
                    '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
                    'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
                    '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
                    'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
                    '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
                    '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
                    'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
                    'V5865', '99662', '28860', '36201', '56210']

###################################
#提取诊断标签，更换为chet重编码的诊断代码
###################################
def extract_diagnosis_labels(diagnoses):
    # 给 diagnoses DataFrame 添加一列 VALUE，并将其值设置为1，表示存在该诊断。
    global diagnosis_labels
    diagnoses['VALUE'] = 1
    #选择 ICUSTAY_ID、ICD9_CODE 和 VALUE 三列，并删除重复项。
    labels = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates()\
                      .pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE').fillna(0).astype(int) #使用 pivot 方法将 ICD9_CODE 作为列，ICUSTAY_ID 作为索引，VALUE 作为值，创建一个新的 DataFrame labels。
    # 遍历 diagnosis_labels 列表，确保 labels DataFrame 包含所有指定的 ICD9 代码，即使它们的值为0（即病人没有此诊断）。
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels]   #确保 labels DataFrame 的列顺序与 diagnosis_labels 一致。
    return labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)




###################################
# Time series preprocessing
# 时间序列数据预处理
###################################

# 读取 itemid 到变量映射 ，从文件中读取 MIMIC-III 数据库中的 itemid 与对应变量名称之间的映射关系。
def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = dataframe_from_csv(fn, index_col=None).fillna('').astype(str) #.fillna(''): 将所有缺失值替换为空字符串。.astype(str): 将所有列的数据类型转换为字符串。
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)   #将 COUNT 列转换为整数类型。
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]   # 筛选出 variable_column 非空且 COUNT 大于 0 的行。
    var_map = var_map[(var_map.STATUS == 'ready')]  #进一步筛选出状态为 "ready" 的行。
    var_map.ITEMID = var_map.ITEMID.astype(int) #将 ITEMID 列转换为整数类型。
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')   #选择特定列并设置 ITEMID 为索引。
    return var_map.rename({variable_column: 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}, axis=1)

# 将事件表 (events) 中的 itemid 转换为对应的变量名，以便后续分析。
# 将 events DataFrame 与 var_map DataFrame 进行合并，基于 ITEMID 列和 var_map 的索引。这一步将 ITEMID 转换为对应的变量名。
def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='ITEMID', right_index=True)

# 从文件中读取每个变量的有效数值范围，包括异常低、有效低、填补值、有效高和异常高。
def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))  # 创建一个字典，用于将列名中的空格替换为下划线。
    to_rename[variable_column] = 'VARIABLE' #将 variable_column 重命名为 VARIABLE。
    var_ranges = dataframe_from_csv(fn, index_col=None)
    # var_ranges = var_ranges[variable_column].apply(lambda s: s.lower())
    var_ranges = var_ranges[columns]    #重命名列名。
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges.set_index('VARIABLE', inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]

# 根据预先定义的范围，移除或调整超出正常范围的数值。
def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = (events.VARIABLE == variable)
    v = events.VALUE[idx].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, 'VALUE'] = v
    return events

# SBP (收缩压)、DBP (舒张压)、CRR (毛细血管再充盈反应) 和 FIO2 (吸入氧气分数)
# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)

# 毛细血管再充盈反应 (CRR) 清理
# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.VALUE is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.VALUE.astype(str)

    v.loc[(df_value_str == 'Normal <3 secs') | (df_value_str == 'Brisk')] = 0
    v.loc[(df_value_str == 'Abnormal >3 secs') | (df_value_str == 'Delayed')] = 1
    return v

# 吸入氧气分数 (FIO2) 清理
# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.VALUE.astype(float).copy()

    ''' The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    ''' The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    '''
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=bool)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))

    v.loc[idx] = v[idx] / 100.
    return v

# 实验室测试值 (GLUCOSE, PH) 清理
# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)

# 氧饱和度 (O2SAT) 清理
# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.loc[idx] = v[idx] * 100.
    return v

# 体温 (Temperature) 清理
# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'F' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v

# 体重 (Weight) 和身高 (Height) 清理
# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.VALUE.astype(float).copy()
    # ounces
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    v.loc[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'lb' in s.lower())
    v.loc[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
clean_fns = {
    'Capillary refill rate': clean_crr,
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    'Fraction inspired oxygen': clean_fio2,
    'Oxygen saturation': clean_o2sat,
    'Glucose': clean_lab,
    'pH': clean_lab,
    'Temperature': clean_temperature,
    'Weight': clean_weight,
    'Height': clean_height
}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.VARIABLE == var_name)
        try:
            events.loc[idx, 'VALUE'] = clean_fn(events[idx])
        except Exception as e:
            import traceback
            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            exit()
    return events.loc[events.VALUE.notnull()]

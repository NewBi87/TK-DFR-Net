import os
from datetime import datetime
from collections import OrderedDict

import pandas
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_intime_col = 'adm_intime'
    adm_outtime_col = 'adm_outtime'
    cid_col = 'cid'

    def __init__(self, path):
        self.path = path

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.admission_procedures = None
        self.admission_medications = None

        self.parse_fn = {'d': self.set_diagnosis}

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    def set_patients(self):
        raise NotImplementedError
    
    @staticmethod
    def to_standard_icd9(code: str):
        raise NotImplementedError

    # 根据SUBJECT_ID合并两个表格。
    def merge_on_subject(table1, table2):
        return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])

    # 根据SUBJECT_ID和HADM_ID合并两个表格。
    def merge_on_subject_admission(table1, table2):
        return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

    # 解析住院记录
    def parse_admission(self):
        print('parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission()
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters) #根据cols指定的列读取需要的数据。
        admissions = self._after_read_admission(admissions, cols)   #调用 _after_read_admission 方法对读取后的数据进行后处理。具体操作因子类的实现不同而有所差异，通常用于进一步筛选或清洗数据。例如，在 Mimic4Parser 中， _after_read_admission 会筛选掉某些无效的住院记录。
        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid, adm_id, adm_intime, adm_outtime = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_intime_col]],row[cols[self.adm_outtime_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({self.adm_id_col: adm_id, self.adm_intime_col: adm_intime, self.adm_outtime_col: adm_outtime,})
        print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:   #只处理住院记录数量大于或等于2的病人。那些住院记录少于2次的病人将被过滤掉。
                patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_intime_col])

        self.patient_admission = patient_admission

    #提供了一个灵活的接口，使得在读取概念数据后可以进行自定义的后处理。
    def _after_read_admission(self, admissions, cols):
        return admissions

    # 主要功能是从指定的CSV文件中解析概念数据（如诊断代码），并将这些数据按住院ID进行组织，返回一个字典，方便后续分析和处理。它通过使用转换器、进度显示和条件过滤，确保解析过程的效率和准确性。
    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()     #确保传入的concept_type是有效的类型，存在于self.parse_fn字典中。如果不在，会引发一个断言错误。
        filename, cols, converters = self.parse_fn[concept_type]()  #根据给定的concept_type，从self.parse_fn中调用相应的方法，以获取文件名、列名和转换器。
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:         #如果设置为跳过病人ID检查，或者当前病人ID存在于已解析的住院记录中，继续处理。这允许一些灵活性，特别是在数据集较大或需要包含所有病人的情况下。
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '':
                    continue
                if adm_id not in result:         #检查result字典中是否已存在该住院ID，如果没有，初始化一个空列表用于存储相关的代码。
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(code)         #将当前的诊断代码（或概念代码）添加到该住院ID的代码列表中。
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        return result
    
    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    #解析诊断代码
    #传入参数 'd'（表示诊断类型），以解析相应的CSV文件。该方法会返回一个字典，包含与住院记录相关的诊断代码，并将其存储在 self.admission_codes 中。
    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        self.admission_codes = self._parse_concept('d')

    #校准病人数据，确保只有那些有有效诊断记录的病人被保留
    def calibrate_patient_by_admission(self):
        print('calibrating patients by admission ...')
        del_pids = []
        #遍历病人记录
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    break
            else:
                continue
            del_pids.append(pid)    #如果某病人ID的所有住院记录都无效，则将其添加到 del_pids 列表中。
        #删除无效病人记录
        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]

    #确保在 self.admission_codes 中仅保留那些与有效病人住院记录相关的住院ID（adm_id），从而清理掉无效的住院记录数据。
    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        #生成一个列表 del_adm_ids，其中包含所有在 self.admission_codes 中但不在 adm_id_set 中的住院ID。这意味着这些住院ID没有对应的有效病人记录。
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        for adm_id in del_adm_ids:
            del self.admission_codes[adm_id]

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def get_pid(self):
        return list(self.patient_admission.keys())

    # 按患者分割诊断记录
    def break_up_admits_by_subject(self):
        subjects = self.get_pid()
        nb_subjects = subjects.shape[0]
        for subject_id in tqdm(subjects, total=nb_subjects, desc= 'Breaking up admission by subjects'):
            dn = os.path.join(self.path,str(subject_id))
            try:
                os.makedirs(dn)
            except:
                pass

    def read_events_table_by_row(self, table):
        nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
        # 打开csv文件并创建DictReader，.upper()确保大写
        reader = csv.DictReader(open(os.path.join(self.path, table.upper() + '.csv'), 'r'))
        # csv.DictReader 是一个迭代器，它会将CSV文件的每一行解析为一个字典，其中键是CSV文件的列名，值是该行对应的列值。
        for i, row in enumerate(reader):
            yield row, i, nb_rows[table.lower()]

    def read_events_table_and_break_up_by_subject(self, table, output_path,
                                                  items_to_keep=None, subjects_to_keep=None):
        # 定义 CSV 文件的列头。
        obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
        if items_to_keep is not None:
            items_to_keep = set([str(s) for s in items_to_keep])
        if subjects_to_keep is not None:
            subjects_to_keep = set([str(s) for s in subjects_to_keep])

        # 定义一个内部类 DataStats，用于存储当前患者的 ID 和观测数据。
        class DataStats(object):
            def __init__(self):
                self.curr_subject_id = ''
                self.curr_obs = []

        data_stats = DataStats()

        def write_current_observations():  # 用于将当前患者的观测数据写入 CSV 文件。
            dn = os.path.join(output_path, str(data_stats.curr_subject_id))
            try:
                os.makedirs(dn)
            except:
                pass
            fn = os.path.join(dn, 'events.csv')
            if not os.path.exists(fn) or not os.path.isfile(fn):
                f = open(fn, 'w')
                f.write(','.join(obs_header) + '\n')
                f.close()
            w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
            w.writerows(data_stats.curr_obs)
            data_stats.curr_obs = []

        nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
        nb_rows = nb_rows_dict[table.lower()]

        for row, row_no, _ in tqdm(self.read_events_table_by_row(table), total=nb_rows,
                                   desc='Processing {} table'.format(table)):

            if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
                continue
            if (items_to_keep is not None) and (row['ITEMID'] not in items_to_keep):
                continue

            row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                       'HADM_ID': row['HADM_ID'],
                       'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                       'CHARTTIME': row['CHARTTIME'],
                       'ITEMID': row['ITEMID'],
                       'VALUE': row['VALUE'],
                       'VALUEUOM': row['VALUEUOM']}
            if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
                write_current_observations()
            data_stats.curr_obs.append(row_out)
            data_stats.curr_subject_id = row['SUBJECT_ID']

        if data_stats.curr_subject_id != '':
            write_current_observations()

    ## patient_admission数据由ADMISSION.csv文件构成(pid,adm_id,adm_time),admission_code数据由DIAGNOSES.csv构成
    def parse(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_diagnoses()
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)

        #提取每个患者的事件文件并按照患者ID划分
        subjects = self.get_pid()
        event_tables = ['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS']
        parent_path = os.path.dirname(self.path)
        output_path = os.path.join(parent_path, 'patients')
        for table in event_tables:
            self.read_events_table_and_break_up_by_subject(table, output_path, subjects_to_keep=subjects)
        return self.patient_admission, self.admission_codes


class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_intime_col: 'ADMITTIME', self.adm_outtime_col:'DISCHTIME', 'diagnosis': 'DIAGNOSIS'}
        converter = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S'),
            'DISCHTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S'),
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': Mimic3Parser.to_standard_icd9}
        return filename, cols, converter

    def set_patients(self):
        filename = 'PATIENTS.csv'
        cols = {
            self.pid_col: 'SUBJECT_ID',
            'gender': 'GENDER',
            'dob': 'DOB',
            'dod': 'DOD'
        }
        converters = {
            'SUBJECT_ID': int,
            'GENDER': lambda x: x,  # 确保 GENDER 字段直接读取为字符串
            'DOB': lambda x: pd.to_datetime(x) if pd.notnull(x) else None,  # 转换日期字段
            'DOD': lambda x: pd.to_datetime(x) if pd.notnull(x) else None  # 转换日期字段
        }
        return filename, cols, converters

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code

class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'Pure Victorian Logical']
        converters = {'ICD10': str, 'Pure Victorian Logical': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['ICD10']: row['Pure Victorian Logical'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_intime_col: 'admittime', self.adm_outtime_col:'dischtime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S'),
            'dischtime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_intime_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return Mimic4Parser.to_standard_icd9(code)

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts

    @staticmethod
    def to_standard_icd9(code: str):
        return Mimic3Parser.to_standard_icd9(code)


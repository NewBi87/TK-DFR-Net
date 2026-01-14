from collections import OrderedDict

from preprocess.parse_csv import EHRParser


def encode_code(patient_admission, admission_codes):
    #构建code_map
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission[EHRParser.adm_id_col]]
            #对于每个诊断代码，如果它还没有被编码，则为其分配一个唯一的整数ID
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map)

    #编码admission_codes，将每个住院记录中的ICD代码替换为对应的整数ID
    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map

import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 配置
CODE_MAP_PATH = '../data/mimic3/encoded/code_map.pkl'  # 确认你的路径
OUTPUT_PATH = '../data/mimic3/standard/pretrained_emb.npy'
MODEL_NAME = "dmis-lab/biobert-v1.1"  # 医学专用 BERT


def generate():
    print("Loading Code Map...")
    with open(CODE_MAP_PATH, 'rb') as f:
        code_map = pickle.load(f)  # {'428.0': 0, ...}

    # 按照 index 顺序提取疾病描述 (ICD9 描述需要你有对应的字典，或者这里直接用 Code 文本代替)
    # 为了简单起见，我们先用 Code 字符串本身做语义 (如 "428.0")
    # *进阶：如果你有 icd9_description 字典，这里替换为描述文本效果更好*
    code_list = sorted(code_map.keys(), key=lambda k: code_map[k])

    print(f"Loading BERT model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embeddings = []
    print("Generating embeddings...")

    with torch.no_grad():
        for i, code_text in enumerate(code_list):
            # 简单的把 '428.0' 变成向量 (如果有疾病全称描述更好)
            text = f"Diagnosis code {code_text}"

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
            outputs = model(**inputs)

            # 取 [CLS] token 的向量作为整个疾病的表示
            emb = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings.append(emb)

            if i % 100 == 0:
                print(f"Processed {i}/{len(code_list)}")

    final_emb = np.array(embeddings)
    np.save(OUTPUT_PATH, final_emb)
    print(f"Saved pretrained embeddings to {OUTPUT_PATH}, shape: {final_emb.shape}")


if __name__ == "__main__":
    generate()
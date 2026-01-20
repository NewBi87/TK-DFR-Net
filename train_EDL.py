import os
import random
import time

import torch
import numpy as np

from models.model import Model
# from models.model import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler, BinaryEDLLoss
from metrics import evaluate_codes, evaluate_hf
from preprocess import load_timeseries_data

def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result

if __name__ == '__main__':
    seed = 6669     # 1337 42 56
    dataset = 'mimic3'  # 'mimic3' or 'mimic4'
    task = 'm'  # 'm' or 'h'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    # device = torch.device('cuda:1' if torch.cuda.is_available() and use_cuda else 'cpu')

    code_size = 48
    graph_size = 48
    hidden_size = 320  # rnn hidden size 150
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 200 #200

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')
    patient_path = os.path.join('data', dataset,'patient_timeseries')

    # code_adj = load_adj(dataset_path, device=device, use_ontology=True, alpha=0.05)
    # code_num = len(code_adj)
    # print("code num is: ",code_num)

    code_adj = load_adj(dataset_path, device=device, use_ontology=True, alpha=0.05)
    # 获取 code_num (取 tuple 第一个元素的长度)
    code_num = len(code_adj[0])
    print("code num is: ", code_num)

    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

    task_conf = {
        'm': {
            'dropout': 0.45,     # 0.45
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }

    # 加载时序数据
    timeseries_data_train = load_timeseries_data(os.path.join(patient_path,"train"), "timeseries_data_train")
    timeseries_data_test = load_timeseries_data(os.path.join(patient_path, "test"), "timeseries_data_test")
    timeseries_data_valid = load_timeseries_data(os.path.join(patient_path, "valid"), "timeseries_data_valid")

    feature_input_dim = 76
    # 模型初始化
    output_size = task_conf[task]['output_size']
    # [修改2、3]移除Sigmoid激活，替换Loss函数
    # activation = torch.nn.Sigmoid()
    # loss_fn = torch.nn.BCELoss()
    #
    # activation = None
    # loss_fn = BinaryEDLLoss(annealing_step=20, device=device)

    # 1. 初始化两个 Loss 函数
    # BCE 用于预训练 (Warm-up)
    bce_loss = torch.nn.BCELoss()
    # EDL 用于微调 (Fine-tune)
    # 此时模型已经收敛，可以给一点 step 让它适应，但不需要太慢
    edl_loss = BinaryEDLLoss(annealing_step=20, device=device)

    valid_bce_loss = torch.nn.BCEWithLogitsLoss()

    # 2. 设置模型激活函数为 None
    # 这样模型输出的就是 Logits，方便我们在训练循环里灵活处理
    activation = None

    # 3. 初始化 Sigmoid (用于 BCE 阶段手动转换)
    sigmoid = torch.nn.Sigmoid()

    # 【修复点 1】定义一个用于评估的 loss_fn
    # 因为 metrics.py 里的 evaluate_codes 接收 Logits，所以这里必须传 edl_loss
    # (或者 BCEWithLogitsLoss)，不能传普通的 BCELoss
    loss_fn = edl_loss

    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    # model = Model(code_num=code_num, code_size=code_size,
    #               adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
    #               t_output_size=t_output_size,
    #               output_size=output_size, dropout_rate=dropout_rate,activation=activation,
    #               feature_input_dim= feature_input_dim,device=device).to(device)

    # Model 初始化时，直接把 tuple 传给 adj 参数
    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj,  # 这里传入 tuple
                  graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation,
                  feature_input_dim=feature_input_dim, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params) #模型中所有可训练参数的总数

    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))

        # --- 核心策略：分阶段 ---
        if epoch < 30:
            if epoch == 0: print(">>> [Phase 1] Warm-up with BCE Loss...")
            current_loss_fn = bce_loss
            use_edl = False
            # 验证时使用 BCEWithLogitsLoss，确保 Loss 数值可读
            current_valid_loss_fn = valid_bce_loss
        else:
            if epoch == 30: print(">>> [Phase 2] Switching to EDL Loss...")
            current_loss_fn = edl_loss
            use_edl = True
            current_loss_fn.update_epoch(epoch - 30)
            # 验证时使用 EDL Loss
            current_valid_loss_fn = edl_loss
        # ---------------------

        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()

        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, divided, y, neighbors, pid_index = train_data[step]

            # 获取 Logits
            output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_train).squeeze()

            if use_edl:
                # Phase 2: 直接计算 EDL Loss
                loss = current_loss_fn(output, y)
            else:
                # Phase 1: Logits -> Sigmoid -> BCELoss
                probs = sigmoid(output)
                loss = current_loss_fn(probs, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

            if (step + 1) % 50 == 0 or step == 0 or (step + 1) == steps:
                end_time = time.time()
                remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
                print('    Step %d / %d, remaining time: %s, loss: %.4f'
                      % (step + 1, steps, remaining_time, total_loss / total_num))

        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))

        save_data = (epoch == epochs - 1)

        # 使用 current_valid_loss_fn 进行验证，确保验证集 Loss 不会报错
        valid_loss, f1_score = evaluate_fn(model, valid_data, current_valid_loss_fn, output_size, timeseries_data_valid,
                                           test_historical, save_data)
        torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))

    print("\nTraining completed. Hyperparameters used:")
    print(f"Seed: {seed}")
    print(f"Dataset: {dataset}")
    print(f"Graph Size: {graph_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"T Attention Size: {t_attention_size}")
    print(f"T Output Size: {t_output_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate Initial: {task_conf[task]['lr']['init_lr']}")
    print(f"Learning Rate Milestones: {task_conf[task]['lr']['milestones']}")
    print(f"Learning Rate Values: {task_conf[task]['lr']['lrs']}")
    print(f"Total Trainable Parameters: {pytorch_total_params}")

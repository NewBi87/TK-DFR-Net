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
            'dropout': 0.5,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                # [核心修改] 将里程碑推迟到 Warmup (50) 之后
                # 策略：
                # 0-60 epoch: LR = 0.01 (BCE预热 + EDL适应期)
                # 60-120 epoch: LR = 1e-3 (EDL精调)
                # 120+ epoch: LR = 1e-4 (收敛)
                'milestones': [60, 120],
                'lrs': [1e-3, 1e-4]
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
    # edl_loss = BinaryEDLLoss(annealing_step=50, device=device)
    edl_loss = BinaryEDLLoss(annealing_step=100, kl_weight=1e-4, device=device)

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

        # [修改点 2] 延长预热时间：从 30 增加到 50 (甚至 100)
        # 建议: 如果总 epoch=200, 预热 50 是比较稳妥的
        WARMUP_EPOCHS = 50

        if epoch < WARMUP_EPOCHS:
            if epoch == 0: print(">>> [Phase 1] Warm-up with BCE Loss...")
            current_loss_fn = bce_loss
            use_edl = False
            current_valid_loss_fn = valid_bce_loss
        else:
            if epoch == WARMUP_EPOCHS: print(">>> [Phase 2] Switching to EDL Loss...")
            current_loss_fn = edl_loss
            use_edl = True

            # [修改点 3] 关键！这里必须减去新的预热长度
            # 否则 annealing 计算会出错 (直接从很大的系数开始)
            current_loss_fn.update_epoch(epoch - WARMUP_EPOCHS)

            current_valid_loss_fn = edl_loss

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

            # [修改 4] 增加梯度裁剪 (Gradient Clipping)
            # 防止 EDL Loss 在初期产生巨大的梯度震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    # [新增] --------- 阶段 10: 自适应阈值搜索 ---------
    print("\n>>> [Phase 10] Starting Threshold Search on Validation Set...")

    # 1. 加载最好的模型 (如果 save_data 逻辑是存最后一个，这里直接用 model 即可；如果是存最好，需重新加载)
    # 假设当前 model 是最后状态，或者您可以加载 param_path 下的 best model
    # 这里我们直接用训练结束后的 model 进行演示
    model.eval()

    y_true_all = []
    y_prob_all = []

    # 2. 在验证集上收集所有的预测概率和真实标签
    with torch.no_grad():
        for step in range(len(valid_data)):
            code_x, visit_lens, divided, y, neighbors, pid_index = valid_data[step]
            output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_valid).squeeze()

            # 确保使用 Sigmoid (如果是 MSE/BCE) 或者 Evidence 转换 (如果是 EDL)
            # 因为我们在 forward 里把 activation 设为了 None，这里需要手动处理
            # 我们的 Loss 是 MSE，所以 output 是 Logits，需要 Sigmoid 变概率
            # 或者是 EDL 的 evidence?
            # 检查 utils.py 的 Loss: MSE 版使用的是 evidence = relu(logits), prob = evidence / (evidence+1)
            # 或者简单的 Sigmoid?
            # 让我们回顾 tk_v3_mse.log 的配置，我们用的是 MSE。
            # 无论是哪种，为了阈值搜索，我们需要 "概率值"。
            # 对于 MSE (Regression)，output 可能就是概率 (如果最后一层有激活)，或者 Logits。
            # 检查 Model: output = model(...) -> Model 最后通常没有激活 (因为 activation=None passed)
            # 所以这里应用 Sigmoid 是通用的安全做法 (将 logits 映射到 0-1)
            probs = torch.sigmoid(output)

            y_true_all.append(y.cpu().numpy())
            y_prob_all.append(probs.cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    # 3. 搜索最佳阈值
    best_thr = 0.5
    best_f1 = 0.0

    from sklearn.metrics import f1_score

    # 搜索区间 [0.1, 0.9], 步长 0.05
    for thr in np.arange(0.1, 0.95, 0.05):
        y_pred_bin = (y_prob_all > thr).astype(int)
        # 计算 Micro F1 (MIMIC-3 标准)
        f1 = f1_score(y_true_all, y_pred_bin, average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"Best Threshold found on Validation: {best_thr:.2f} (F1: {best_f1:.4f})")

    # 4. 用最佳阈值在测试集上重新评估
    print(f">>> Applying Best Threshold ({best_thr:.2f}) to Test Set...")

    y_true_test = []
    y_prob_test = []

    with torch.no_grad():
        for step in range(len(test_data)):
            code_x, visit_lens, divided, y, neighbors, pid_index = test_data[step]
            output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_test).squeeze()
            probs = torch.sigmoid(output)
            y_true_test.append(y.cpu().numpy())
            y_prob_test.append(probs.cpu().numpy())

    y_true_test = np.concatenate(y_true_test, axis=0)
    y_prob_test = np.concatenate(y_prob_test, axis=0)

    y_pred_test = (y_prob_test > best_thr).astype(int)
    final_f1 = f1_score(y_true_test, y_pred_test, average='micro')

    print(f"Final Test F1 Score (with thr={best_thr:.2f}): {final_f1:.4f}")
    if final_f1 > 0.2505:
        print("🎉 Congratulations! You have beaten the Baseline (0.2505)!")
    else:
        print(f"Gap to Baseline: {0.2505 - final_f1:.4f}")

        # ... (在打印出 Final Test F1 之后) ...

        print("\n>>> [Phase 11] Detailed Subgroup Analysis with Optimized Threshold...")
        # 我们调用 metrics.py 里的评估函数，但是要稍微改一下让它支持传入 threshold
        # 或者我们直接在这里手动算一下 Visit 分组结果

        # 简单的实现方式：
        # 遍历 test_data，用 best_thr 算 F1
        visit_stats = {}  # key: visit_num, value: [y_true, y_pred]

        with torch.no_grad():
            for step in range(len(test_data)):
                code_x, visit_lens, divided, y, neighbors, pid_index = test_data[step]
                output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_test).squeeze()
                probs = torch.sigmoid(output)

                # 使用最佳阈值
                pred_binary = (probs > best_thr).float()

                # 记录按 Visit 分组的数据
                for i in range(len(code_x)):
                    v_num = visit_lens[i]  # 这是一个 tensor 或 int
                    if isinstance(v_num, torch.Tensor): v_num = v_num.item()

                    if v_num not in visit_stats:
                        visit_stats[v_num] = {'true': [], 'pred': []}
                    visit_stats[v_num]['true'].append(y[i].cpu().numpy())
                    visit_stats[v_num]['pred'].append(pred_binary[i].cpu().numpy())

        print(f"\nSubgroup Performance at Threshold = {best_thr:.2f}:")
        for v_num in sorted(visit_stats.keys()):
            yt = np.array(visit_stats[v_num]['true'])
            yp = np.array(visit_stats[v_num]['pred'])
            if len(yt) > 0:
                sub_f1 = f1_score(yt, yp, average='micro')
                print(f"  Visit = {v_num} (n={len(yt)}): F1 = {sub_f1:.4f}")
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

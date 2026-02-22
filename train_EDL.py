import os
import random
import time

import torch
import numpy as np

from models.model import Model
# from models.model import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler, BinaryEDLLoss, historical_hot
from metrics import evaluate_codes, evaluate_hf, top_k_prec_recall
from preprocess import load_timeseries_data

# def historical_hot(code_x, code_num, lens):
#     result = np.zeros((len(code_x), code_num), dtype=int)
#     for i, (x, l) in enumerate(zip(code_x, lens)):
#         result[i] = x[l - 1]
#     return result



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

    # edl_loss = BinaryEDLLoss(annealing_step=100, kl_weight=1e-4, device=device)

    # [修改] 初始化混合 Loss (Focal Loss 版)
    # aux_weight=0.5: 让 Focal Loss 有足够力度去修正 Top-K 排序
    edl_loss = BinaryEDLLoss(annealing_step=100, kl_weight=1e-4, device=device)
    # bce_loss 仅用于评估打印，保持不变
    bce_loss = torch.nn.BCELoss()
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

    # ... (Phase 10 代码保持不变) ...
    print(f"Best Threshold found on Validation: {best_thr:.2f} (F1: {best_f1:.4f})")

    # ... (Phase 10 代码保持不变) ...

    # --------- 阶段 11: 全面评估 (Phase 11: Comprehensive Evaluation) ---------
    print(f"\n>>> [Phase 11] Comprehensive Evaluation (Top-K & Best Thr)...")

    # [导入模块] 确保引入了所需的评估函数
    from metrics import top_k_prec_recall, calculate_occurred
    from utils import historical_hot
    from sklearn.metrics import f1_score

    # [Metric 1] Standard Top-K Metrics (Baseline Comparison)
    # 这部分保持不变，使用 dummy_loss 绕过
    dummy_loss = lambda x, y: torch.tensor(0.0).to(x.device)
    real_test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    print("\n[Metric 1] Standard Top-K Metrics (Rank-based - Baseline Comparison):")
    model.eval()
    evaluate_fn(model, test_data, dummy_loss, output_size, timeseries_data_test, real_test_historical, False)

    # [Metric 2] Optimized Threshold Metrics (Detailed Subgroup Analysis)
    print(f"\n[Metric 2] Optimized Threshold Metrics (Thr={best_thr:.2f}):")

    visit_stats = {}

    # 用于收集全局数据
    all_y_true = []
    all_y_sorted = [] # 排序后的索引 (用于 Ranking 指标)
    all_y_pred_bin = [] # 阈值后的 0/1 (用于 F1)
    all_hist = []     # 历史数据 (用于 Occurred)

    with torch.no_grad():
        for step in range(len(test_data)):
            code_x, visit_lens, divided, y, neighbors, pid_index = test_data[step]
            output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_test).squeeze()
            probs = torch.sigmoid(output)

            y_np = y.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # 1. 计算 Ranking 用的排序索引 (降序)
            # argsort 默认升序，所以用 [:, ::-1] 翻转
            sorted_preds = np.argsort(probs_np, axis=-1)[:, ::-1]

            # 2. 计算 F1 用的二值预测 (基于最佳阈值)
            pred_bin = (probs_np > best_thr).astype(int)

            # 3. 计算 Occurred 用的历史数据
            hist_np = historical_hot(code_x, code_num, visit_lens)

            # 收集全局数据
            all_y_true.append(y_np)
            all_y_sorted.append(sorted_preds)
            all_y_pred_bin.append(pred_bin)
            all_hist.append(hist_np)

            # 收集分组数据
            for i in range(len(code_x)):
                v_num = visit_lens[i]
                if isinstance(v_num, torch.Tensor): v_num = v_num.item()
                if v_num not in visit_stats:
                    visit_stats[v_num] = {
                        'y_true': [], 'y_sorted': [], 'y_pred_bin': [], 'hist': []
                    }

                visit_stats[v_num]['y_true'].append(y_np[i])
                visit_stats[v_num]['y_sorted'].append(sorted_preds[i])
                visit_stats[v_num]['y_pred_bin'].append(pred_bin[i])
                visit_stats[v_num]['hist'].append(hist_np[i])

    # --- 辅助函数：打印全面指标 ---
    def print_comprehensive_metrics(y_true, y_sorted, y_pred_bin, hist, prefix=""):
        # 1. 计算 F1 (使用阈值后的结果)
        # 注意：这里我们用 sklearn 的 micro F1，对应你要求的 f1_score
        f1_val = f1_score(y_true, y_pred_bin, average='micro')

        # 2. 计算 Ranking 指标 (Precision, Recall)
        ks = [10, 20, 30, 40]
        prec_list, recall_list = top_k_prec_recall(y_true, y_sorted, ks)

        # 3. 计算 Occurred 指标
        r1, r2 = calculate_occurred(hist, y_true, y_sorted, ks)

        # 4. 格式化输出 (完全匹配你的要求)
        print(
            '%s f1_score: %.4f --- top_k_precision: %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
            % (
                prefix, f1_val,
                prec_list[0], prec_list[1], prec_list[2], prec_list[3],
                recall_list[0], recall_list[1], recall_list[2], recall_list[3],
                r1[0], r1[1], r1[2], r1[3],
                r2[0], r2[1], r2[2], r2[3]
            )
        )

    # 1. 打印全局指标
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_sorted = np.concatenate(all_y_sorted, axis=0)
    all_y_pred_bin = np.concatenate(all_y_pred_bin, axis=0)
    all_hist = np.concatenate(all_hist, axis=0)

    print(f"Overall (Thr={best_thr:.2f}):")
    print_comprehensive_metrics(all_y_true, all_y_sorted, all_y_pred_bin, all_hist, prefix="Evaluation:")

    # 2. 打印分组指标
    print("-" * 120) # 加长分隔线以适应长输出
    for v_num in sorted(visit_stats.keys()):
        data = visit_stats[v_num]
        yt = np.array(data['y_true'])

        if len(yt) == 0: continue

        ys = np.array(data['y_sorted'])
        yb = np.array(data['y_pred_bin'])
        yh = np.array(data['hist'])

        print(f"Visit {v_num} (n={len(yt)}):")
        print_comprehensive_metrics(yt, ys, yb, yh, prefix="  ")
        print("-" * 30)

    print("="*60)

    # # --------- 阶段 12: 不确定性案例分析 (严谨版) ---------
    # print(f"\n>>> [Phase 12] Exporting Case Studies for Qualitative Analysis...")
    #
    # model.eval()
    # cases = []
    #
    # with torch.no_grad():
    #     for step in range(len(test_data)):
    #         code_x, visit_lens, divided, y, neighbors, pid_index = test_data[step]
    #         output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_test).squeeze()
    #
    #         evidence = torch.relu(output)
    #         S = evidence + 2
    #         uncertainty = 2 / S
    #         probs = (evidence + 1) / S
    #         preds = (probs > best_thr).float()
    #
    #         y_np = y.cpu().numpy()
    #         u_np = uncertainty.cpu().numpy()
    #         preds_np = preds.cpu().numpy()
    #
    #         for i in range(len(code_x)):
    #             # 计算样本级 F1 和 平均不确定性
    #             p_f1 = f1_score(y_np[i], preds_np[i], average='binary')
    #             mean_u = np.mean(u_np[i])
    #             cases.append({
    #                 'pid_index': pid_index[i],
    #                 'f1': p_f1,
    #                 'mean_uncertainty': mean_u
    #             })
    #
    # # [核心修改] 放弃固定阈值，改用基于真实分布的排名 (Top 5% vs Bottom 20%)
    # # 1. 按照 F1 分数降序排列所有病例
    # cases.sort(key=lambda x: x['f1'], reverse=True)
    #
    # # 2. 提取 Top 50 作为 Good Cases，Bottom 200 作为 Bad Cases
    # # (具体数字可根据 test_data 总样本量调整，保证不为 0)
    # top_n = min(50, len(cases) // 10)
    # bottom_n = min(200, len(cases) // 3)
    #
    # good_cases = cases[:top_n]
    # bad_cases = cases[-bottom_n:]
    #
    # print(
    #     f"  Extracted Top {len(good_cases)} Genuine Good Cases (Avg F1: {np.mean([c['f1'] for c in good_cases]):.4f})")
    # print(
    #     f"  Extracted Bottom {len(bad_cases)} Genuine Bad Cases (Avg F1: {np.mean([c['f1'] for c in bad_cases]):.4f})")
    #
    # if len(good_cases) > 0 and len(bad_cases) > 0:
    #     avg_u_good = np.mean([c['mean_uncertainty'] for c in good_cases])
    #     avg_u_bad = np.mean([c['mean_uncertainty'] for c in bad_cases])
    #
    #     print(f"\n[Hypothesis Verification] Uncertainty vs Performance:")
    #     print(f"  Avg Uncertainty (Good Predictions): {avg_u_good:.4f}")
    #     print(f"  Avg Uncertainty (Bad Predictions):  {avg_u_bad:.4f}")
    #
    #     if avg_u_bad > avg_u_good:
    #         diff_pct = (avg_u_bad - avg_u_good) / avg_u_good * 100
    #         print(f"SUCCESS: Uncertainty increased by {diff_pct:.2f}% on hard cases.")
    #     else:
    #         print(" WARNING: Uncertainty is not correlated with error.")
    #
    #     # 严谨打印真实的 Best Case
    #     best_case = good_cases[0]
    #     print(f"\n[Genuine Case Study]")
    #     print(
    #         f"  Best Case (PID {best_case['pid_index']}): Actual F1={best_case['f1']:.4f}, U={best_case['mean_uncertainty']:.4f}")
    # else:
    #     print(" Evaluation failed: Not enough valid cases.")
    #
    # print("=" * 60)
    # --------- 阶段 12: 精确 Case Study 筛选与轻量化导出 ---------
    # --------- 阶段 12: 精确 Case Study 筛选与轻量化导出 ---------
    print(f"\n>>> [Phase 12] Exporting Case Studies for Qualitative Analysis...")

    model.eval()
    success_candidates = []
    failure_candidates = []

    with torch.no_grad():
        for step in range(len(test_data)):
            code_x, visit_lens, divided, y, neighbors, pid_index = test_data[step]
            output = model(code_x, divided, neighbors, visit_lens, pid_index, timeseries_data_test).squeeze()

            # EDL 核心计算
            evidence = torch.relu(output)
            S = evidence + 2
            uncertainty = 2 / S
            probs = (evidence + 1) / S

            y_np = y.cpu().numpy()
            u_np = uncertainty.cpu().numpy()
            probs_np = probs.cpu().numpy()

            for i in range(len(code_x)):
                v_len = visit_lens[i]
                if isinstance(v_len, torch.Tensor): v_len = v_len.item()

                # 获取该患者真实的疾病索引
                true_idx = np.where(y_np[i] == 1)[0].tolist()
                true_num = len(true_idx)

                if true_num == 0:
                    continue

                # 【核心修复】：彻底废弃 best_thr，对齐 metrics.py 的 Top-K (K=true_num) 排序提取法
                # 按照 EDL 输出概率从高到低排序，取最自信的前 true_num 个疾病作为预测结果
                sorted_idx = np.argsort(probs_np[i])[::-1]
                pred_idx = sorted_idx[:true_num].tolist()

                # 生成用于计算当前样本个体 F1 的二进制数组
                pred_bin = np.zeros_like(y_np[i])
                pred_bin[pred_idx] = 1

                # 此时算出的 p_f1 将完美符合全局日志的分布规律
                p_f1 = f1_score(y_np[i], pred_bin, average='binary', zero_division=0)
                mean_u = np.mean(u_np[i])

                case_data = {
                    'pid': pid_index[i],
                    'visit_count': v_len,
                    'f1': p_f1,
                    'uncertainty': mean_u,
                    'history_codes': code_x[i][:v_len - 1],
                    'current_true': true_idx,
                    'current_pred': pred_idx,
                    'uncertainty_map': u_np[i]
                }

                # 筛选成功案例 (门槛：就诊>=2次，由于TopK法则提取更严格，F1>=0.30即可抓到极其精彩的病例)
                if v_len >= 2 and p_f1 >= 0.30:
                    success_candidates.append(case_data)

                # 筛选失败拦截案例 (门槛：就诊>=2次，预测完全失准 F1=0，且不确定性>0.85)
                elif v_len >= 2 and p_f1 < 0.05 and mean_u > 0.85:
                    failure_candidates.append(case_data)

    # 排序并截断 (各取 Top 50 保证轻量化)
    success_candidates.sort(key=lambda x: x['f1'], reverse=True)
    failure_candidates.sort(key=lambda x: x['uncertainty'], reverse=True)

    final_success = success_candidates[:50]
    final_failure = failure_candidates[:50]

    import pickle

    with open('case_study_results.pkl', 'wb') as f:
        pickle.dump({'success': final_success, 'failure': final_failure}, f)

    print(f"  -> 成功提取并保存 {len(final_success)} 个优质 Success Cases")
    print(f"  -> 成功提取并保存 {len(final_failure)} 个高不确定性 Failure Cases")
    print("=" * 60)

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

import torch
import numpy as np
import pickle
from sklearn.metrics import f1_score, roc_auc_score

# 评估函数
#计算 F1 分数，用于评估多标签分类任务的性能。
def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)

#: 计算 Top-K 精确率（Precision@K）和召回率（Recall@K），用于评估多标签分类任务的性能。
def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            # r[i] += len(it) / min(k, len(t))
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)

#计算预测结果中历史发生过的标签和未发生过的标签的正确率，用于评估模型对历史信息的利用情况。
def calculate_occurred(historical, y, preds, ks):
    # y_occurred = np.sum(np.logical_and(historical, y), axis=-1)
    # y_prec = np.mean(y_occurred / np.sum(y, axis=-1))
    r1 = np.zeros((len(ks), ))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    for i, k in enumerate(ks):
        # n_k = np.minimum(n, k)
        n_k = n
        pred_k = np.zeros_like(y)
        for T in range(len(pred_k)):
            pred_k[T][preds[T][:k]] = 1
        # pred_occurred = np.sum(np.logical_and(historical, pred_k), axis=-1)
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)
        r1[i] = np.mean(np.sum(pred_occurred_true, axis=-1) / n_k)
        r2[i] = np.mean(np.sum(pred_not_occurred_true, axis=-1) / n_k)
    return r1, r2

#评估模型在诊断代码预测任务上的性能，包括损失、F1 分数、Top-K 召回率以及历史发生率。
# def evaluate_codes(model, dataset, loss_fn, output_size, timeseries_data, historical=None, save_data=False):
#     model.eval()                    #设置为评估模式
#     total_loss = 0.0
#     labels = dataset.label()        #获取真实标签，EHRDataset 的 label() 方法返回整个数据集的标签数组，类型为ndarray：，形状为：[patient_num, label_nums]
#     preds = []   #用于收集每个批次的预测结果
#     patient_id = []
#     for step in range(len(dataset)):                                    #EHRDataset 实现了 __len__ 方法，返回数据集中的批次数。
#         code_x, visit_lens, divided, y, neighbors, pids = dataset[step]       #获取该批次数据
#
#         output = model(code_x, divided, neighbors, visit_lens, pids, timeseries_data)          #获取模型对当前批次的预测，使用模型对当前批次的数据进行预测。EHRDataset 提供的 code_x、divided、neighbors 和 visit_lens 是模型输入所需的各种特征。
#         pred = torch.argsort(output, dim=-1, descending=True)           #对模型输出按类别维度进行排序，pred的形状是[batch_size, num_labels],tensor类型
#         preds.append(pred)                                              #添加预测结果,长度等于数据集中批次数目，而每个元素都是一个形状为 [batch_size, num_labels] 的张量列表list。
#         patient_id.append((pids))
#         loss = loss_fn(output, y)                                       #计算损失
#         total_loss += loss.item() * output_size * len(code_x)           #累积当前批次的损失值到total_loss中。这里乘以output_size和len(code_x)是为了考虑到批次大小和输出维度的影响。然而，这种累积方式可能会导致总损失值过大，通常只需要loss.item() * len(code_x)即可。
#         print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
#     avg_loss = total_loss / dataset.size()                              #计算平均损失
#     preds = torch.vstack(preds).detach().cpu().numpy()
#     f1_score = f1(labels, preds)                                        #调用前面定义的f1函数计算F1分数。
#     prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])    #调用top_k_prec_recall函数计算Top-K精确率和召回率。
#     # if historical is not None:                                          #如果提供了历史标签信息，则继续执行以下代码块。
#     #     r1, r2 = calculate_occurred(historical, labels, preds, ks=[10, 20, 30, 40]) #调用calculate_occurred函数计算历史发生过的标签和未发生过的标签的正确率。
#     #     print('\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
#     #           % (avg_loss, f1_score, recall[0], recall[1], recall[2], recall[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3]))
#     # else:
#     #     print('\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
#     #           % (avg_loss, f1_score, recall[0], recall[1], recall[2], recall[3]))
#
#     if historical is not None:
#         r1, r2 = calculate_occurred(historical, labels, preds, ks=[10, 20, 30, 40])
#         print(
#             '\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_precision: %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
#             % (
#             avg_loss, f1_score, prec[0], prec[1], prec[2], prec[3], recall[0], recall[1], recall[2], recall[3], r1[0],
#             r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3]))
#     else:
#         print(
#             '\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_precision: %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
#             % (avg_loss, f1_score, prec[0], prec[1], prec[2], prec[3], recall[0], recall[1], recall[2], recall[3]))
#
#     if save_data:
#         patient_id = np.concatenate(patient_id)
#         actual_labels = [np.where(row == 1)[0].tolist() for row in labels]
#
#         np.save('preds.npy', preds)
#         np.save('patient_id.npy', patient_id)
#         with open('actual_labels.pkl', 'wb') as f:
#             pickle.dump(actual_labels, f)
#
#     #return avg_loss, f1_score       #返回平均损失和F1分数，作为函数的输出。
#     # 调用分层分析函数
#     subgroup_analysis_by_visits(dataset, labels, preds)
#     return avg_loss, f1_score  # 返回平均损失和F1分数，作为函数的输出。



def evaluate_codes(model, dataset, loss_fn, output_size, timeseries_data, historical=None, save_data=False):
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    preds = []
    patient_id = []

    # [关键] 只要还在评估阶段，不需要计算梯度，节省显存
    with torch.no_grad():
        for step in range(len(dataset)):
            code_x, visit_lens, divided, y, neighbors, pids = dataset[step]

            # 1. 获取模型输出 (Logits)
            logits = model(code_x, divided, neighbors, visit_lens, pids, timeseries_data)

            # 2. 计算损失 (Loss函数内部会处理Logits)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * output_size * len(code_x)

            # 3. [核心修改] 将 Logits 转为 概率 (Probability) 用于评估
            # 虽然 argsort 对 Logits 和 Prob 结果一样，但如果以后有阈值判断，必须用 Prob
            # 且有些 Loss 计算可能需要 Prob
            output = torch.sigmoid(logits)

            pred = torch.argsort(output, dim=-1, descending=True)
            preds.append(pred)
            patient_id.append((pids))

            print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')

    avg_loss = total_loss / dataset.size()
    preds = torch.vstack(preds).detach().cpu().numpy()
    f1_score = f1(labels, preds)
    prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])

    if historical is not None:
        r1, r2 = calculate_occurred(historical, labels, preds, ks=[10, 20, 30, 40])
        print(
            '\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_precision: %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
            % (
                avg_loss, f1_score, prec[0], prec[1], prec[2], prec[3], recall[0], recall[1], recall[2], recall[3],
                r1[0],
                r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3]))
    else:
        print(
            '\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_precision: %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
            % (avg_loss, f1_score, prec[0], prec[1], prec[2], prec[3], recall[0], recall[1], recall[2], recall[3]))

    if save_data:
        patient_id = np.concatenate(patient_id)
        actual_labels = [np.where(row == 1)[0].tolist() for row in labels]

        np.save('preds.npy', preds)
        np.save('patient_id.npy', patient_id)
        with open('actual_labels.pkl', 'wb') as f:
            pickle.dump(actual_labels, f)

    subgroup_analysis_by_visits(dataset, labels, preds)
    return avg_loss, f1_score


def subgroup_analysis_by_visits(dataset, labels, preds):
    """按就诊次数分层计算指标（英文输出版）"""
    # 获取每个患者的就诊次数（dataset.visit_lens是列表，每个元素是该患者的就诊次数）
    visit_counts = dataset.visit_lens
    # 获取所有可能的就诊次数（如1次、2次、3次等）
    unique_counts = sorted(list(set(visit_counts)))
    print("\n    Subgroup analysis by number of visits:")  # 替换为英文

    for count in unique_counts:
        # 筛选出就诊次数为count的患者索引
        mask = [i for i, v in enumerate(visit_counts) if v == count]
        if not mask:  # 避免空列表
            continue
        # 提取该组的真实标签和预测结果
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        # 计算该组的指标
        sub_f1 = f1(sub_labels, sub_preds)
        sub_prec, sub_recall = top_k_prec_recall(sub_labels, sub_preds, ks=[10, 20, 30, 40])
        # 打印结果（全英文，解决乱码）
        print(f"    Number of visits = {count} (total {len(mask)} patients):")
        print(f"        F1 score: {sub_f1:.4f}")
        print(f"        Precision@10: {sub_prec[0]:.4f}, @20: {sub_prec[1]:.4f}")
        print(f"        Recall@10: {sub_recall[0]:.4f}, @20: {sub_recall[1]:.4f}")

#评估模型在心力衰竭预测任务上的性能，包括损失、AUC 和 F1 分数。
def evaluate_hf(model, dataset, loss_fn, output_size=1, historical=None):
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    outputs = []
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
        output = model(code_x, divided, neighbors, visit_lens).squeeze()
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        output = output.detach().cpu().numpy()
        outputs.append(output)
        pred = (output > 0.5).astype(int)
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    avg_loss = total_loss / dataset.size()
    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)
    auc = roc_auc_score(labels, outputs)
    f1_score_ = f1_score(labels, preds)
    print('\r    Evaluation: loss: %.4f --- auc: %.4f --- f1_score: %.4f' % (avg_loss, auc, f1_score_))
    return avg_loss, f1_score_


# metrics.py

def evaluate_mc_dropout(model, dataset, timeseries_data, num_samples=10):
    """
    MC Dropout 不确定性评估
    原理：在推理阶段保持 Dropout 开启，对每个样本预测 num_samples 次。
    均值 -> 最终预测结果
    方差 -> 不确定性 (Uncertainty)
    """
    print(f"\n>>> Starting MC Dropout Evaluation (Samples: {num_samples})...")

    # 【关键】开启 Dropout 层，但冻结 BatchNorm 层
    # model.train() 会开启 Dropout 和 BN
    # 我们只想要 Dropout，所以手动处理一下
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()  # 强制开启 Dropout

    all_preds = []

    # 只需要跑一部分数据验证一下即可，不用全跑
    # 这里为了演示，我们只取 dataset 的第一个 batch
    with torch.no_grad():
        step = 0
        code_x, visit_lens, divided, y, neighbors, pids = dataset[step]

        # 对同一个 batch 重复预测 N 次
        batch_outputs = []
        for i in range(num_samples):
            # 获取 Logits -> Prob
            logits = model(code_x, divided, neighbors, visit_lens, pids, timeseries_data)
            probs = torch.sigmoid(logits)
            batch_outputs.append(probs.cpu().numpy())  # [batch, num_labels]

        # stack 起来: [num_samples, batch, num_labels]
        batch_outputs = np.array(batch_outputs)

        # 计算均值 (预测值) 和 方差 (不确定性)
        mean_preds = np.mean(batch_outputs, axis=0)
        uncertainty = np.var(batch_outputs, axis=0)

        # 打印一下看看效果
        print(f"Sample 0 Prediction: {mean_preds[0][:10]}")  # 打印前10个标签的概率
        print(f"Sample 0 Uncertainty: {uncertainty[0][:10]}")  # 打印前10个标签的不确定性
        print(f"Max Uncertainty in Batch: {np.max(uncertainty)}")
        print(f"Min Uncertainty in Batch: {np.min(uncertainty)}")

    return mean_preds, uncertainty

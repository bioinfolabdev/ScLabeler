import random
from collections import Counter

import pandas as pd
import torch.nn.functional as F
import joblib
import torch
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_random_state, shuffle
from imblearn.over_sampling import SMOTE
import time
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
import os
import dataprocess
from dataprocess import prepare_graph_data


def evaluate(species_name, tissue_name, label_mapping, model, data_loader, criterion, unassigned_threshold,
             is_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_cells = 0

    # 检查是否存在 Unrecognized 类别
    unrecognized_label = None
    if 'Unrecognized' in label_mapping:
        unrecognized_label = label_mapping['Unrecognized']

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            outputs, _ , _ = model(batch_data)
            loss = criterion(outputs, batch_data.y)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            # 处理未分配的情况
            unassigned_mask = max_probs < unassigned_threshold
            if unrecognized_label is not None:
                 preds[unassigned_mask] = unrecognized_label  # 将未分配的样本标记为 Unrecognized
            else:
                 preds[unassigned_mask] = -1  # 如果没有 Unrecognized 类别，标记为 -1

            total_cells += batch_data.y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_data.y.cpu().numpy())

    average_loss = total_loss / len(data_loader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if is_test:
        # 计算每个类别的样本数量和准确度
        label_to_accuracy = {}
        label_to_count = {}

        for label in np.unique(all_labels):
            label_mask = all_labels == label
            label_count = label_mask.sum()

            if label_count > 0:
                accuracy = accuracy_score(all_labels[label_mask], all_preds[label_mask])
            else:
                accuracy = 0.0

            label_to_count[label] = label_count
            label_to_accuracy[label] = accuracy

        # 转换类别标签名称
        cell_types = {v: k for k, v in label_mapping.items()}

        results = {
            "Cell Type": [cell_types[l] for l in label_to_count.keys()],
            "Sample Count": list(label_to_count.values()),
            "Accuracy": list(label_to_accuracy.values())
        }

        # 创建 DataFrame 并保存为 CSV 文件
        df = pd.DataFrame(results)
        # 根据 'Sample Count' 倒序排序
        df = df.sort_values(by='Sample Count', ascending=False)
        output_dir = "train results/Specific results/result"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{species_name}_{tissue_name}_class_distribution.csv"
        output_path = os.path.join(output_dir, output_filename)

        df.to_csv(output_path, index=False)

        print(f"Results saved to {output_path}")

        # 计算未分配样本
        unassigned_count = (all_preds == unrecognized_label).sum() if unrecognized_label is not None else (
                all_preds == -1).sum()

        # assigned_labels 和 assigned_preds
        assigned_labels = all_labels
        assigned_preds = all_preds

        if len(assigned_labels) > 0:
            accuracy = accuracy_score(assigned_labels, assigned_preds)
            f1 = f1_score(assigned_labels, assigned_preds, average='weighted', zero_division=0)
            macro_f1 = f1_score(assigned_labels, assigned_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(assigned_labels, assigned_preds, average='micro', zero_division=0)
        else:
            accuracy = f1 = macro_f1 = micro_f1 = 0.0

        # 计算未分配率
        unassigned_rate = unassigned_count / len(all_labels)
        unassigned_rate = 0
        metrics = {
            'test_cells': total_cells,
            'accuracy': accuracy,
            'f1_score': f1,
            'macro_f1_score': macro_f1,
            'micro_f1_score': micro_f1,
            'unassigned_rate': unassigned_rate
        }
    else:
        # 在验证集上计算包含未分配样本的准确率
        accuracy = accuracy_score(all_labels, all_preds)
        metrics = {
            'accuracy': accuracy
        }

    return average_loss, metrics


def train_and_evaluate(species_name, tissue_name, label_mapping, train_loader, val_loader, model, optimizer, criterion,
                       best_model_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_accuracy = 0.0
    patience_counter = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(batch_data)
            loss = criterion(outputs, batch_data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_metrics = evaluate(species_name, tissue_name, label_mapping, model, val_loader, criterion,
                                         config.unassigned_threshold, is_test=False)
        val_accuracy = val_metrics['accuracy']
        val_accuracies.append(val_accuracy)
        train_losses.append(total_loss / len(train_loader))

        if epoch % 50 == 0:
            print(
                f'Epoch {epoch}: Train Loss {total_loss / len(train_loader):.4f}, Val Accuracy {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"No improvement in {patience_counter} epochs, stopping early.")
            break

    return best_val_accuracy


# 在每个数据集上合并样本不足的类别
def merge_small_classes(X, y, threshold, unrecognized_label, min_unrecognized_samples):
    counter = Counter(y)
    excluded_labels = [label for label, count in counter.items() if count < threshold]
    filtered_labels = [label for label, count in counter.items() if count >= threshold]

    # 检查是否有足够的样本合并为 Unrecognized
    unrecognized_sample_count = sum(count for label, count in counter.items() if label in excluded_labels)

    if unrecognized_sample_count >= min_unrecognized_samples:
        # 创建新的标签列表，将少于阈值的类别替换为Unrecognized
        y_merged = np.array([unrecognized_label if label in excluded_labels else label for label in y])
        print(f"合并以下类别为 'Unrecognized': {excluded_labels}")
    else:
        # 如果没有足够样本，则直接舍弃这些类别
        y_merged = np.array([label for label in y if label in filtered_labels])
        X = X[[label in filtered_labels for label in y]]
        print(f"舍弃以下少量类别: {excluded_labels}")

    return X, y_merged


def run_experiment(species_name, tissue_name, config):
    # 确保保存模型的目录存在
    os.makedirs('model', exist_ok=True)
    os.makedirs('predict', exist_ok=True)
    train_data_path = "train_data"
    save_path = dataprocess.create_species_tissue_path(species_name, tissue_name)
    best_model_path = os.path.join(save_path, 'best_model.pth')
    gene_list_path = os.path.join(save_path, 'gene_list.npy')
    label_mapping_path = os.path.join(save_path, 'label_mapping.joblib')

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Step 1: 收集和预处理数据
    gene_expression, labels, label_mapping = dataprocess.load_preprocessed_data(train_data_path, species_name,
                                                                                tissue_name)

    # 获取细胞类型名称
    label_mapping = dict(sorted(label_mapping.items(), key=lambda item: item[1]))  # 假设 label_mapping 是从名称到原始数值标签的映射
    cell_types = list(label_mapping.keys())  # 细胞类型名称列表

    # 打印细胞类型名称和对应的原始标签
    print("数据中包含的细胞类别")
    for key, value in label_mapping.items():
        print(f"{value}: {key}")

    # Step 2: 准备 gene_expression 和 labels
    np.save(gene_list_path, gene_expression.index)
    gene_expression = gene_expression.T.to_numpy()
    gene_expression, labels = shuffle(gene_expression, labels, random_state=seed)

    # Step 3: 定义类别数阈值和 unrecognized_label
    total_samples = len(labels)
    base_threshold_ratio = 500
    dynamic_threshold = total_samples // base_threshold_ratio
    cell_threshold = max(20, min(50, dynamic_threshold))  # 定义多数类的样本数阈值
    unrecognized_label = len(cell_types)  # 使用新的标签编号表示 Unrecognized 类

    # Step 4: 将类别分为多数类和少数类
    counter = Counter(labels)
    majority_labels = {label for label, count in counter.items() if count >= cell_threshold}

    # 初始化最终数据集
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    # Step 5: 处理多数类和少数类，合并后将少数类映射为 Unrecognized 类
    # 处理多数类样本分割
    for label in majority_labels:
        label_samples = [(sample, label) for sample, lbl in zip(gene_expression, labels) if lbl == label]
        if len(label_samples) > 0:
            X_lbl, y_lbl = zip(*label_samples)
            X_train_val_lbl, X_test_lbl, y_train_val_lbl, y_test_lbl = train_test_split(X_lbl, y_lbl,
                                                                                        test_size=config.test_size,
                                                                                        random_state=seed)
            X_train_lbl, X_val_lbl, y_train_lbl, y_val_lbl = train_test_split(X_train_val_lbl, y_train_val_lbl,
                                                                              test_size=config.val_size,
                                                                              random_state=seed)
            X_train.extend(X_train_lbl)
            y_train.extend(y_train_lbl)
            X_val.extend(X_val_lbl)
            y_val.extend(y_val_lbl)
            X_test.extend(X_test_lbl)
            y_test.extend(y_test_lbl)

    # Step 6: 检查少数类是否有任何样本数量 >= 5，并且少数类的总数大于 cell_threshold
    minority_count_total = sum(count for label, count in counter.items() if label not in majority_labels and count >= 3)
    has_eligible_minority = minority_count_total >= cell_threshold and any(
        count >= 5 for label, count in counter.items() if label not in majority_labels)

    # 如果存在符合条件的少数类，则处理少数类
    if has_eligible_minority:
        # 处理少数类样本分割
        for label, count in counter.items():
            if label not in majority_labels:
                label_samples = [(sample, label) for sample, lbl in zip(gene_expression, labels) if lbl == label]
                X_lbl, y_lbl = zip(*label_samples)
                if count < 3:
                    continue  # 跳过此标签
                elif 3 <= count <= 4:
                    # 先将样本分为训练集和验证集，验证集比例为 1 个样本
                    if count == 3:
                        test_size = 1 / 3  # 1 个样本分配给验证集
                    else:  # count == 4
                        test_size = 0.25  # 1 个样本分配给验证集
                    X_train_lbl, X_val_lbl, y_train_lbl, y_val_lbl = train_test_split(X_lbl, y_lbl, test_size=test_size,
                                                                                      random_state=seed)
                    X_train.extend(X_train_lbl)
                    y_train.extend([unrecognized_label] * len(X_train_lbl))
                    X_val.extend(X_val_lbl)
                    y_val.extend([unrecognized_label] * len(X_val_lbl))

                elif count >= 5:
                    X_train_val_lbl, X_test_lbl, y_train_val_lbl, y_test_lbl = train_test_split(X_lbl, y_lbl,
                                                                                                test_size=0.2,
                                                                                                random_state=seed)
                    X_train_lbl, X_val_lbl, y_train_val_lbl, y_val_lbl = train_test_split(X_train_val_lbl,
                                                                                          y_train_val_lbl,
                                                                                          test_size=0.2,
                                                                                          random_state=seed)
                    X_train.extend(X_train_lbl)
                    y_train.extend([unrecognized_label] * len(X_train_lbl))
                    X_val.extend(X_val_lbl)
                    y_val.extend([unrecognized_label] * len(X_val_lbl))
                    X_test.extend(X_test_lbl)
                    y_test.extend([unrecognized_label] * len(X_test_lbl))

    # Step 6: 转换为 numpy 数组，准备后续处理
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Step 7: 创建新的连续标签映射，使用细胞类型名称
    all_labels = set(y_train) | set(y_val) | set(y_test)
    valid_cell_types = [cell_types[label] for label in all_labels if label != unrecognized_label]
    if unrecognized_label in all_labels:
        valid_cell_types.append("Unrecognized")
    # 生成从细胞类型名称到连续整数的映射
    label_mapping = {cell_type: idx for idx, cell_type in enumerate(valid_cell_types)}
    n_classes = len(label_mapping)
    joblib.dump(label_mapping, label_mapping_path)

    # 打印并验证新的标签映射
    print()
    print("重新标识的细胞类别")
    for key, value in label_mapping.items():
        print(f"{value}: {key}")

    # Step 8: 重新映射标签，使其连续
    # 通过新的 label_mapping 映射更新 y_train, y_val, y_test
    y_train = np.array(
        [label_mapping[cell_types[label] if label != unrecognized_label else "Unrecognized"] for label in y_train])
    y_val = np.array(
        [label_mapping[cell_types[label] if label != unrecognized_label else "Unrecognized"] for label in y_val])
    y_test = np.array(
        [label_mapping[cell_types[label] if label != unrecognized_label else "Unrecognized"] for label in y_test])

    class_distribution = dict(sorted(Counter(y_train).items()))
    print()
    print(f"Original train class distribution: {class_distribution}")
    # 设定扩充后的总样本目标
    # target_total_samples = 100000
    # num_classes = len(class_distribution)
    # 动态计算每个类别的最大扩充数量
    # max_samples_per_class = target_total_samples // num_classes
    max_class_count = max(class_distribution.values())
    threshold = max_class_count / 3
    # target_samples = {
    #     class_label: max(count, min(int(max_class_count / 2), max_samples_per_class) if count < threshold else count)
    #     for class_label, count in class_distribution.items()
    # }
    # 设置 SMOTE 采样策略，不对“未知”类别应用 SMOTE
    target_samples = {
        class_label: int(max_class_count / 2) if count < threshold and class_label != unrecognized_label else count
        for class_label, count in class_distribution.items()
    }
    # #
    # 获取最小类别的样本数量
    min_class_count = min(class_distribution.values())
    # 设置 k_neighbors，不大于最小类的样本数 - 1
    k_neighbors = max(2, min(min_class_count - 1, 7))  # 设定下限为 2
    smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=target_samples, random_state=check_random_state(seed))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    new_distribution = dict(sorted(Counter(y_train_resampled).items()))
    print("New train class distribution after SMOTE:", new_distribution)
    print()

    classes = np.unique(y_train_resampled)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_resampled)
    # 找到“未知”类别的索引（假设 unrecognized_label 定义了“未知”类别的标签）
    if unrecognized_label in classes:
        unrecognized_index = np.where(classes == unrecognized_label)[0][0]
        class_weights[unrecognized_index] *= 1.2  # 比其他类别稍高，可根据需求调整
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_data = prepare_graph_data(X_train_resampled, y_train_resampled, k=5)
    val_data = prepare_graph_data(X_val, y_val, k=5)
    test_data = prepare_graph_data(X_test, y_test, k=5)

    train_data.x = dataprocess.add_noise_to_features(train_data.x)
    train_data.edge_index, train_data.edge_attr = dataprocess.perturb_graph_structure(train_data.edge_index,
                                                                                      train_data.num_nodes,
                                                                                      train_data.edge_attr)

    num_features = train_data.num_features
    num_classes = len(set(train_data.y.numpy()))
    
    if 'Unrecognized' in label_mapping:
    # 如果有 Unrecognized 类别，置信度阈值基于已知类别数量（不包含 Unrecognized）
        num = len(label_mapping) - 1
    else:
        # 如果没有 Unrecognized 类别，置信度阈值基于所有类别
        num = len(label_mapping)

    config.unassigned_threshold=1/num+0.05
    print(config.unassigned_threshold)
    model = dataprocess.GNNWithAttention(num_features, num_classes, num_heads=config.num_heads,
                                         hidden_dim=config.hidden_dim).to(device)
    # model = dataprocess.GNN(num_features, num_classes, dropout_rate=0.5).to(device)
    optimizer_name = config.optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    train_loader = DataLoader([train_data], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader([test_data], batch_size=config.batch_size, shuffle=False)

    # 训练和评估模型
    best_val_accuracy = train_and_evaluate(species_name, tissue_name, label_mapping, train_loader, val_loader, model,
                                           optimizer,
                                           criterion, best_model_path, config)

    end_time = time.time()
    Time = end_time - start_time
    print()
    print(f"Training completed in {Time: .2f} seconds.")
    print(f"Best model accuracy: {best_val_accuracy:.4f}")

    # 在测试集上进行最终评估，不包含未分配样本
    model = torch.load(best_model_path)
    test_loss, test_metrics = evaluate(species_name, tissue_name, label_mapping, model, test_loader, criterion,
                                       config.unassigned_threshold, is_test=True)
    return total_samples, test_metrics['test_cells'], test_metrics['accuracy'], test_metrics['f1_score'], test_metrics[
        'macro_f1_score'], \
        test_metrics['micro_f1_score'], test_metrics['unassigned_rate'], len(label_mapping), Time

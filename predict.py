import os
import time
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score

import dataprocess
from dataprocess import prepare_graph_data

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 将 4 替换为您想要使用的核心数
def load_mapping_table(mapping_table_path):
    """
    Load the mapping table and create a dictionary for mapping Original_Type to Mapped_Type.
    """
    mapping_table = pd.read_excel(mapping_table_path, sheet_name='Sheet1')
    return dict(zip(mapping_table['Original_name'], mapping_table['Cell-type']))

def standardize_and_map(cell_type, mapping_dict):
    """
    使用映射字典标准化并映射细胞类型。
    如果输入的细胞类型在映射字典中未找到，返回 "Other"。

    参数:
        cell_type (str): 要标准化和映射的细胞类型。
        mapping_dict (dict): 一个字典，将原始细胞类型映射到标准化的类型。

    返回:
        str: 如果找到匹配类型，返回标准化的细胞类型；否则返回 "Other"。
    """
    if not isinstance(cell_type, str) or cell_type.strip() == "":
        return "Other"  # 处理空或无效的细胞类型
    return mapping_dict.get(cell_type.strip(), "Other")

# 更新匹配逻辑，考虑末尾 s 的情况
def match_with_s_variants(cell_type, label_mapping, unrecognized_label):
    """
    匹配真实细胞类型，考虑末尾 s 的两种情况。
    """
    if cell_type in label_mapping:
        return label_mapping[cell_type]
    elif cell_type.rstrip('s') in label_mapping:
        return label_mapping[cell_type.rstrip('s')]
    elif cell_type + 's' in label_mapping:
        return label_mapping[cell_type + 's']
    else:
        return unrecognized_label  # 未匹配到的映射为 Unrecognized 的标签值

def process_file_pair(species_name, tissue_name, data_file, celltype_file, model_folder, save_folder, mapping_dict):
    """
    Process each file pair, predict cell types, and compute accuracy, F1 Score, ARI, and NMI.
    """
    save_path = os.path.join(model_folder, species_name, tissue_name)
    print(f"Loading model from: {save_path}")

    # Load model components
    gene_list_path = os.path.join(save_path, 'gene_list.npy')
    label_mapping_path = os.path.join(save_path, 'label_mapping.joblib')
    gene_list = np.load(gene_list_path, allow_pickle=True)
    label_mapping = joblib.load(label_mapping_path)

    start_time = time.time()
    # Load model
    best_model_path = os.path.join(save_path, 'best_model.pth')
    model = torch.load(best_model_path)
    model.eval()

    # Preprocess data
    gene_expression, valid_cell_names = dataprocess.prediction_data_preprocess(data_file, gene_list)
    gene_expression = gene_expression.T.to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load true labels
    true_labels = None
    if os.path.exists(celltype_file):
        true_labels_df = pd.read_csv(celltype_file, sep='\t', header=0)
        true_cell_types = true_labels_df.iloc[:, 1]
        valid_true_cell_types = true_cell_types.loc[true_labels_df.iloc[:, 0].isin(valid_cell_names)]

        # 过滤掉 "Uncharacterized" 的真实标签数据
        valid_indices = valid_true_cell_types != "Uncharacterized"
        # 过滤掉 "Uncharacterized" 的真实标签数据
        valid_true_cell_types = valid_true_cell_types[valid_indices]
        # 同步过滤掉基因表达矩阵中的对应细胞
        # 将 valid_indices 转换为布尔列表以便与 valid_cell_names 一起使用
        valid_indices_list = valid_indices.tolist()
        # 同步过滤掉基因表达矩阵中的对应细胞
        valid_cell_names = [name for name, keep in zip(valid_cell_names, valid_indices_list) if keep]
        gene_expression = gene_expression[valid_indices, :]

        # 使用映射表将真实细胞类型标准化
        valid_true_cell_types = valid_true_cell_types.map(lambda x: standardize_and_map(x, mapping_dict))

        # 处理未能映射的值，将其标记为 "Unrecognized"
        valid_true_cell_types = valid_true_cell_types.fillna("Unrecognized")

        # 对 label_mapping 的键进行标准化处理
        standardized_label_mapping = {}
        for key, value in label_mapping.items():
            standardized_key = standardize_and_map(key, mapping_dict)
            if standardized_key not in standardized_label_mapping:
                standardized_label_mapping[standardized_key] = value

        # 使用标准化后的 label_mapping 进行标签映射
        label_mapping = standardized_label_mapping

        # 获取 Unrecognized 的标签值
        unrecognized_label = label_mapping.get("Unrecognized", -1)  # 默认值为 -1，如果未定义

        # 将标准化后的类型映射为数值标签
        true_labels_numeric = valid_true_cell_types.map(
            lambda x: match_with_s_variants(x, label_mapping, unrecognized_label))

        # 检查未映射的真实细胞类型
        unmapped_true_types = valid_true_cell_types[true_labels_numeric == unrecognized_label]
        if not unmapped_true_types.empty:
            print(f"Warning: The following true cell types could not be mapped: {set(unmapped_true_types)}")

        # 转换为 PyTorch 张量
        true_labels = torch.tensor(true_labels_numeric.astype(int).values, dtype=torch.long).to(device)

    # Prepare graph data for prediction
    data = prepare_graph_data(gene_expression, labels=None, k=5)
    data = data.to(device)
    if 'Unrecognized' in label_mapping:
    # 如果有 Unrecognized 类别，置信度阈值基于已知类别数量（不包含 Unrecognized）
        num_classes = len(label_mapping) - 1
    else:
        # 如果没有 Unrecognized 类别，置信度阈值基于所有类别
        num_classes = len(label_mapping)
    confidence_threshold = 1 / num_classes + 0.05
    # 获取 Unrecognized 的标签值
    unrecognized_label = label_mapping.get("Unrecognized", -1)  # 默认值为 -1

    # Prediction process

    with torch.no_grad():
        logits, _, _ = model(data)
        # logits = model(data)
        probs = torch.softmax(logits, dim=1)
        max_probs, pred = torch.max(probs, dim=1)
    pred[max_probs < confidence_threshold] = unrecognized_label
    # 计算未分配率
    unassigned_count = (pred == unrecognized_label).sum().item()
    unassigned_rate = round((unassigned_count / len(pred)) * 100, 2)
    end_time = time.time()  # 记录结束时间

    # 计算预测时间
    predict_time = round(end_time - start_time, 2)

    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    pred_cell_types = [inverse_label_mapping.get(p.item(), "Unrecognized") for p in pred]

    # Create results DataFrame
    df = pd.DataFrame({
        'Cell Names': valid_cell_names,
        'Predicted Cell Types': pred_cell_types
    })

    if true_labels is not None:
        true_cell_types_mapped = valid_true_cell_types.values
        df['True Cell Types'] = true_cell_types_mapped
        # 计算未知类型细胞数量
        true_unrecognized_count = (true_labels == unrecognized_label).sum().item()
        # 计算本身为未知类型并且实际也被预测为未知类型的比率
        correctly_predicted_unrecognized_count = (
                    (true_labels == unrecognized_label) & (pred == unrecognized_label)).sum().item()
        correctly_predicted_unrecognized_rate = round(
            (correctly_predicted_unrecognized_count / true_unrecognized_count) * 100,
            2) if true_unrecognized_count > 0 else 0.0

        # 计算其他指标 (例如 Accuracy 和 F1 Score)
        accuracy = round(pred.eq(true_labels).sum().item() / len(true_labels) * 100, 2)  # 转为百分比并保留两位小数
        f1 = round(f1_score(true_labels.cpu().numpy(), pred.cpu().numpy(), average='weighted', zero_division=0) * 100,
                   2)
        ari = round(adjusted_rand_score(true_labels.cpu().numpy(), pred.cpu().numpy()) * 100, 2)
        nmi = round(normalized_mutual_info_score(true_labels.cpu().numpy(), pred.cpu().numpy()) * 100, 2)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"ARI: {ari:.2f}")
        print(f"NMI: {nmi:.2f}")
        print(f"Unknown Cell Count: {true_unrecognized_count}")
        print(f"Correctly Predicted Unknown Rate: {correctly_predicted_unrecognized_rate:.2f}")
    else:
        accuracy = 0.0
        f1 = 0.0
        ari = 0.0
        nmi = 0.0
        true_unrecognized_count = 0
        correctly_predicted_unrecognized_rate = 0
    # Save prediction results
    total_cell = len(pred)
    result_folder = os.path.join(save_folder, species_name, tissue_name)
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, f"{species_name}_{tissue_name}{total_cell}_predictions.csv")
    df.to_csv(result_file, index=False)
    print(f"Predictions saved to {result_file}")

    # 返回结果
    return {
        'Species': species_name,
        'Tissue': tissue_name,
        'Total Cells': len(pred),
        'Accuracy(%)': accuracy,
        'F1 Score(%)': f1,
        'ARI(%)': ari,
        'NMI(%)': nmi,
        'Unassigned Rate (%)': unassigned_rate,
        'Unknown Cell Count': true_unrecognized_count,
        'Correctly Predicted Unknown Rate (%)': correctly_predicted_unrecognized_rate,
        'Prediction Time (s)': predict_time
    }

def find_file_pairs_and_process(test_data_folder, model_folder, save_folder, mapping_table_path):
    """
    Recursively find all file pairs and process them.
    """
    mapping_dict = load_mapping_table(mapping_table_path)
    all_results = []
    for root, dirs, files in os.walk(test_data_folder):
        species_name = os.path.basename(os.path.dirname(root))  # Extract species name
        tissue_name = os.path.basename(root)  # Extract tissue name
        data_file = None
        celltype_file = None
        for file in files:
            if file.endswith('_Data.txt'):
                data_file = os.path.join(root, file)
            elif file.endswith('_Celltype.txt'):
                celltype_file = os.path.join(root, file)

            if data_file and celltype_file:
                print(f"Processing {species_name} - {tissue_name}")
                result = process_file_pair(species_name, tissue_name, data_file, celltype_file, model_folder,
                                           save_folder, mapping_dict)
                all_results.append(result)
                data_file = None
                celltype_file = None

    # Save all results to a summary file
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(save_folder, 'all_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"All results saved to {results_file}")

# Paths
model_folder = 'model'
save_folder = 'results'
test_data_folder = 'test data'
mapping_table_path = 'Celltype/Celltype.xlsx'

# Run the processing function
find_file_pairs_and_process(test_data_folder, model_folder, save_folder, mapping_table_path)


# # 文件路径
# data_file = 'test data/human/Liver/human_Liver3502_Data.txt'
# celltype_file = 'test data/human/Liver/human_Liver3502_Celltype.txt'

# # 相关路径
# species_name = "human"  # 物种名称
# tissue_name = "Liver"  # 组织名称
# model_folder = 'model'  # 模型保存路径
# save_folder = 'results'  # 保存结果的路径
# mapping_table_path = 'Celltype/Celltype.xlsx'  # 映射表路径

# # 加载映射表
# mapping_dict = load_mapping_table(mapping_table_path)

# # 调用 process_file_pair 处理单个文件
# result = process_file_pair(
#     species_name=species_name,
#     tissue_name=tissue_name,
#     data_file=data_file,
#     celltype_file=celltype_file,
#     model_folder=model_folder,
#     save_folder=save_folder,
#     mapping_dict=mapping_dict
# )

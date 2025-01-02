import csv
import os
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 将 4 替换为你的 CPU 核心数量
import joblib
import pandas as pd

from Train import run_experiment



# 定义训练数据文件夹路径
folder_path = 'train_data/human/liver'


class Config:
    test_size = 0.2  # 测试集分割比例
    val_size = 0.2  # 验证集比例
    k = 5  # 图节点最近邻个数
    unassigned_threshold = 0  # 置信度，类别概率低于此值的设定为未标记
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 1000
    batch_size = 512  # 假设使用批处理
    patience = 60
    optimizer = 'adam'  # 'adam'  'adamw' 'rmsprop'
    num_heads = 8  # GAT 注意力头数
    hidden_dim = 8  # 单个注意力头数维度


# 使用 os.path.split 分割路径
parent_dir, last_dir = os.path.split(folder_path)

# 再次分割 parent_dir 获取物种名
species_name = os.path.basename(parent_dir)
tissue_name = last_dir
print(f"Species Name: {species_name}")
print(f"Tissue Name: {tissue_name}")

files = os.listdir(folder_path)

# 细胞类别数
label_mapping = joblib.load(os.path.join(folder_path, f"{species_name}_{tissue_name}_label_mapping.pkl"))
num_cell_types = len(label_mapping)

total_cell_count, test_cells, accuracy, f1_score, macro_f1_score, micro_f1_score, unassigned_rate, trained_celltype, time = \
    run_experiment(species_name, tissue_name, Config)

print()

print(f"Total number of cells: {total_cell_count}")
print(f"test_cells: {test_cells}")
print(f"total cell type：{num_cell_types}")
print(f"trained cell type: {trained_celltype}")
print(f"accuracy: {accuracy}")
print(f"f1_score: {f1_score}")
print(f"macro_f1_score: {macro_f1_score}")
print(f"micro_f1_score: {micro_f1_score}")
print(f"unassigned_rate: {unassigned_rate}")
print(f"train seconds: {time}")

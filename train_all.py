import os
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 将 4 替换为你的 CPU 核心数量
import joblib
import pandas as pd
from Train import run_experiment

# 定义数据文件夹的根路径
base_folder_path = 'train_data'
output_file = 'train result/results.csv'

class Config:
    test_size = 0.2  # 测试集分割比例
    val_size = 0.2  # 验证集比例
    k = 5  # 图节点最近邻个数
    unassigned_threshold = 0
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 500
    batch_size = 512  # 假设使用批处理
    patience = 60
    optimizer = 'adam'  # 选择优化器: 'adam', 'adamw', 'rmsprop'
    num_heads = 8  # GAT 注意力头数
    hidden_dim = 8  # 单个注意力头的维度


# 初始化字典来存储结果
results = []

# 遍历每个物种目录
for species_name in os.listdir(base_folder_path):
    species_path = os.path.join(base_folder_path, species_name)
          # 遍历每个组织目录
    for tissue_name in os.listdir(species_path):
            tissue_path = os.path.join(species_path, tissue_name)

            if os.path.isdir(tissue_path):
                folder_path = tissue_path
                print(f"Species Name: {species_name}")
                print(f"Tissue Name: {tissue_name}")

                # 细胞类别数
                files = os.listdir(folder_path)
                label_mapping = joblib.load(
                    os.path.join(folder_path, f"{species_name}_{tissue_name}_label_mapping.pkl"))
                num_cell_types = len(label_mapping)

                total_cell_count, test_cells, accuracy, f1_score, macro_f1_score, micro_f1_score, unassigned_rate, trained_celltype, time = run_experiment(
                        species_name, tissue_name,
                        Config)
                # 存储结果
                results.append({
                    'species_name': species_name,
                    'tissue_name': tissue_name,
                    'total cells': total_cell_count,
                    'test cells': test_cells,
                    'cell types': num_cell_types,
                    'trained cell types': trained_celltype,
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'macro_f1_score': macro_f1_score,
                    'micro_f1_score': micro_f1_score,
                    'Unassigned Rate': unassigned_rate,
                    'training seconds': time
                })
            print()
# 将结果转换为 DataFrame
df_results = pd.DataFrame(results)

# 保存到 CSV 文件
df_results.to_csv(output_file, index=False)

print(f"结果已保存到 {output_file}")

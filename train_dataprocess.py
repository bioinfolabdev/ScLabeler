import os
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 将 4 替换为你的 CPU 核心数量
import pandas as pd
from Train import run_experiment
from dataprocess import preprocess_and_save

# 定义数据文件夹的根路径
base_folder_path = 'data'
output_path = "train_data"
# 遍历每个物种目录
for species_name in os.listdir(base_folder_path):
    species_path = os.path.join(base_folder_path, species_name)

    if os.path.isdir(species_path):
        # 遍历每个组织目录
        for tissue_name in os.listdir(species_path):
            tissue_path = os.path.join(species_path, tissue_name)

            if os.path.isdir(tissue_path):
                files = os.listdir(tissue_path)
                # 初始化变量来存储文件对
                data_file = None
                celltype_file = None
                file_pairs = []

                # 查找文件对
                for file_name in files:
                    if '_Celltype' in file_name:
                        celltype_file = file_name
                    elif '_Data' in file_name:
                        data_file = file_name

                    # 如果找到了数据文件和细胞类型文件，添加到 file_pairs
                    if data_file and celltype_file:
                        file_pairs.append(
                            (os.path.join(tissue_path, data_file), os.path.join(tissue_path, celltype_file)))

                        data_file = None
                        celltype_file = None

                # 执行实验并收集指标
                if file_pairs:
                    preprocess_and_save(species_name, tissue_name, file_pairs, output_path)

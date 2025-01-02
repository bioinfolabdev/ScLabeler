# ScLabeler

[![python 3.11](https://img.shields.io/badge/python-3.11-brightgreen)](https://www.python.org/) 

### 基于图注意力网络(GAT)的细胞类型注释方法
我们提出了一种基于图注意力网络（Graph Attention Network, GAT）的单细胞类型注释方法ScLabeler。该方法通过引入自适应的注意力机制，在高效捕捉细胞间复杂关系的同时，显著提高了注释的准确性和模型的可解释性。为了进一步验证模型的通用性，我们从多个公开数据库收集了包括人类、小鼠和斑马鱼三种物种的数据，构成了丰富的单细胞数据集。

源码在windows系统，cuda-11.8上运行。

# Install

[![scipy-1.14.1](https://img.shields.io/badge/scipy-1.14.1-yellowgreen)](https://github.com/scipy/scipy) [![torch-2.3.0](https://img.shields.io/badge/torch-1.6.0-orange)](https://github.com/pytorch/pytorch) [![torch-geometric-2.6.1](https://img.shields.io/badge/torchgeometric-2.6.1-red)](https://github.com/pyg-team/pytorch_geometric) [![pandas-2.2.3](https://img.shields.io/badge/pandas-2.2.3-lightgrey)](https://github.com/pandas-dev/pandas) [![pyglib-0.4.0](https://img.shields.io/badge/pyglib-0.4.0-blue)](https://github.com/pyg-team/pytorch_geometric) [![scikit__learn-1.6.0](https://img.shields.io/badge/scikit__learn-1.6.0-green)](https://github.com/scikit-learn/scikit-learn) [![joblib-1.4.2](https://img.shields.io/badge/joblib-1.4.2-yellow)](https://github.com/joblib/joblib)


```

```

# Usage
#### 数据预处理
对单细胞转录组学 csv 数据文件进行预处理，首先根据 NCBI 基因数据库修改基因符号，删除不匹配的基因和重复的基因。并采用Seurat的归一化函数。确保将基因表达矩阵`**_data.csv`和细胞类别文件`**_celltype.csv`（可选）按示例结构放置在[`original data`](https://github.com/bioinfolabdev/ScLabeler/tree/main/original%20data/human/liver)文件夹内，并运行`preprocess.py`，可自动生成预处理数据存放在data文件夹下（基因表达矩阵和细胞类别文件需要前缀一致）。有需要可在预处理后，使用[`Celltype/Generate cell types.py`](https://github.com/bioinfolabdev/ScLabeler/blob/main/Celltype/Generate%20cell%20types.py)生成细胞类型映射表。

#### 使用预训练模型
在`predict.py`文件中指定模型目录`model_folder`、保存目录`save_folder`、待预测数据目录`test_data_folder`以及细胞类型映射文件`mapping_table_path`并运行，可在`results`文件夹下生成汇总表格以及各文件的预测结果：
```
model_folder = 'model'
save_folder = 'results'
test_data_folder = 'test data'
mapping_table_path = 'Celltype/Celltype.xlsx'
```
#### 训练自己的模型
在`train_dataprocess.py`中指定待训练数据目录`base_folder_path`及输出目录`output_path`，进行训练数据的进一步预处理：
```
base_folder_path = 'data'
output_path = "train_data"
```

在`train_all.py`中指定预处理训练数据目录`base_folder_path`及结果输出目录`output_file`，并运行，可在[`model`](https://github.com/bioinfolabdev/ScLabeler/tree/main/model/human/Liver)生成对应模型文件`best_model.pth`、基因列表`gene_list.npy`、以及类别映射字典`label_mapping.joblib`,在输出目录`output_file`下生成模型对各个物种组织的各细胞类别的具体训练准确率：
```
base_folder_path = 'train_data'
output_file = 'train result/results.csv'
```

此后再根据上述使用预训练模型步骤进行预测即可。

#### 可视化
[![R 4.42](https://img.shields.io/badge/R-%3E4.40-blue)](https://www.r-project.org/)  [![pyecharts-2.0.7](https://img.shields.io/badge/tpyecharts-2.0.7-orange)](https://github.com/pyecharts/pyecharts/)

在[`visualization/UMAP.R`](https://github.com/bioinfolabdev/ScLabeler/blob/main/visualization/UMAP.R)中指定基因表达矩阵`expression_matrix`以及细胞预测结果文件路径`sclabeler_file`，可绘制细胞UMAP图：
```
# 1. 读取基因表达矩阵
expression_matrix <- read.table(
  "test data/human/Liver/human_Liver5105_Data.txt",
  header = TRUE, row.names = 1
)

# 2. 细胞预测结果文件路径
sclabeler_file <- "results/human/Liver/human_Liver5101_predictions.csv"
```

| True Cell Types    | Predicted Cell Types |
|-----------------------|-----------------------|
| ![True Cell Types](https://github.com/bioinfolabdev/ScLabeler/blob/main/visualization/True_Cell_Types.png) | ![Predicted Cell Types](https://github.com/bioinfolabdev/ScLabeler/blob/main/visualization/Predicted_Cell_Types.png) |


在[`visualization/Sankey.py`](https://github.com/bioinfolabdev/ScLabeler/blob/main/visualization/Sankey.py)中指定预测数据路径`file_path`，可绘制细胞类型Sankey图：
```
# 读取数据
file_path = '../results/human/Liver/human_Liver5101_predictions.csv'  # 替换为实际文件路径
```


<div style="text-align: center;">
  <a href="https://raw.githubusercontent.com/bioinfolabdev/ScLabeler/main/visualization/human_Liver5101_sankey.html" download>
    <img src="https://github.com/bioinfolabdev/ScLabeler/blob/main/visualization/%E7%BB%86%E8%83%9E%E7%B1%BB%E5%9E%8B%E7%BB%9F%E8%AE%A1.png" alt="Predicted Cell Types" width="700">
  </a>
  <p><strong>点击图片下载 HTML 文件</strong></p>
</div>



# Data availability 
[![scDeepSort-Python](https://img.shields.io/badge/ScLabeler-Python-brightgreen)](1) 

All pre-processed data are available in the form of readily-for-analysis for researchers to develop new methods. Please refer to the release page called [`Pre-processed data`](1)

# Cite

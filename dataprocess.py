import os
import warnings
from collections import OrderedDict

import joblib
import scanpy as sc
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import Dropout, ELU, Linear
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv, BatchNorm, GATConv
import torch.nn.functional as F


def seurat_normalize(expression):
    """
    使用Seurat NormalizeData方法归一化表达数据。

    参数:
    expression (np.ndarray): 基因表达矩阵

    返回:
    np.ndarray: 归一化后的表达矩阵
    """
    adata = sc.AnnData(expression.T)  # 转置以符合AnnData格式
    sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化每个细胞的总表达量
    sc.pp.log1p(adata)  # 对数据取对数
    return adata.X.T  # 转置回来


def quality_control_and_filter(adata):
    """
    质量控制和过滤步骤。

    参数:
    adata (AnnData): 单细胞数据对象

    返回:
    AnnData: 过滤后的单细胞数据对象
    """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    return adata


def load_cell_type_mapping(celltype_file):
    """
    加载统一的细胞类型映射文件。

    参数:
    celltype_file (str): 细胞类型文件路径。

    返回:
    dict: 生成的细胞总型到亚型的映射字典。
    """
    celltype_mapping = pd.read_excel(celltype_file)
    mapping_dict = pd.Series(celltype_mapping['Cell-type'].values, index=celltype_mapping['Cell-subtype']).to_dict()
    return mapping_dict


# 预处理预测数据的函数
def prediction_data_preprocess(file_path, gene_list):
    """
    预处理预测数据并对齐基因列表。

    参数:
    file_path (str): 预测数据的文件路径。
    gene_list (list): 训练时使用的基因列表。

    返回:
    np.array: 对齐后的基因表达数据数组。
    list: 有效的细胞索引。
    """
    df = pd.read_csv(file_path, sep='\t', header=0)
    gene_expression = df.T
    gene_expression.columns = gene_expression.iloc[0]
    gene_expression = gene_expression.drop(gene_expression.index[0])

    # 确保数据格式正确并对齐基因
    expression_matrix = gene_expression.values.astype(float)
    adata = sc.AnnData(X=expression_matrix)
    adata.var_names = gene_expression.columns
    adata.obs_names = gene_expression.index

    adata.X = adata.X.astype('float32')
    # 执行质量控制和标准化
    adata = quality_control_and_filter(adata)
    valid_cell_names = adata.obs_names.to_list()

    normalized_expression = seurat_normalize(adata.X.T)
    expression_df = pd.DataFrame(normalized_expression.T, columns=adata.var_names, index=adata.obs_names)
    expression_df = expression_df.T
    aligned_gene_expression = align_gene_expression(expression_df, gene_list)

    return aligned_gene_expression, valid_cell_names


def train_data_preprocess(species_name, file_pairs=None):
    """
    预处理训练数据并对齐基因列表。

    参数:
    file_pairs (list): 包含标准基因表达文件和细胞类型文件的文件路径对列表。

    返回:
    tuple: (对齐后的基因表达矩阵, 标签数组, 标签映射字典)
    """
    print("data processing…")
    all_data = []
    all_genes = set()
    label_mapping = {}
    batch_indices = []

    if file_pairs is not None:
        for i, (gene_expression_file, cell_type_file) in enumerate(file_pairs):
            expression_df, labels = preprocess_standard_files(gene_expression_file, cell_type_file, label_mapping)
            all_genes.update(expression_df.index)
            all_data.append((expression_df.astype(np.float32), labels))
            batch_indices.extend([i] * expression_df.shape[1])
    else:
        raise ValueError("file_pairs must be provided")

    # 对齐所有数据的基因顺序
    all_genes = sorted(all_genes)
    aligned_data = []
    combined_labels = []

    for expression_df, labels in all_data:
        aligned_gene_expression = align_gene_expression(expression_df, all_genes)
        aligned_data.append(aligned_gene_expression)
        combined_labels.extend(labels)

    # 在 concat 之前将数据类型转换为 float32
    aligned_data = [df.astype('float32') for df in aligned_data]
    combined_expression = pd.concat(aligned_data, axis=1)
    # 确保观测值名称唯一性
    # 为每个细胞名称添加唯一标识符，例如批次号
    combined_expression.columns = [f"{col}_{batch_indices[i]}" for i, col in enumerate(combined_expression.columns)]
    combined_labels = np.array(combined_labels)

    # 忽略 Pandas 的 FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning)
    # 加载数据到 AnnData 对象
    adata = sc.AnnData(X=combined_expression.T)
    adata.obs_names_make_unique()  # 确保观测值名称唯一
    adata.obs['labels'] = combined_labels
    adata.obs['batch'] = pd.Categorical(batch_indices)

    # 选择高变异基因
    sc.pp.highly_variable_genes(adata, batch_key='batch', n_top_genes=6000)

    # 筛选出高变异基因
    adata = adata[:, adata.var.highly_variable]

    # 更新combined_expression为只包含高变异基因的数据
    combined_expression = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

    print("data process complete")
    return combined_expression, combined_labels, label_mapping


def preprocess_and_save(species_name, tissue_name, file_pairs, output_path):
    """
    预处理数据并保存合并后的结果，避免重复计算。

    参数:
    species_name (str): 物种名称。
    tissue_name (str): 组织名称。
    file_pairs (list): 包含基因表达和细胞类型文件的文件路径对。
    output_path (str): 合并数据保存的主路径。
    """
    # 定义保存路径，按物种和组织分级
    save_path = os.path.join(output_path, species_name, tissue_name)
    os.makedirs(save_path, exist_ok=True)

    combined_expression, combined_labels, label_mapping = train_data_preprocess(species_name, file_pairs)

    # 保存数据，文件名带上组织名称
    combined_expression.to_csv(os.path.join(save_path, f"{species_name}_{tissue_name}_combined_expression.csv"))
    np.save(os.path.join(save_path, f"{species_name}_{tissue_name}_combined_labels.npy"), combined_labels)
    joblib.dump(label_mapping, os.path.join(save_path, f"{species_name}_{tissue_name}_label_mapping.pkl"))

    print(f"预处理训练数据已保存到 {save_path}")
    print()


def load_preprocessed_data(output_path, species_name, tissue_name):
    """
    从保存的文件中加载预处理数据。

    参数:
    output_path (str): 保存数据的路径。
    species_name (str): 物种名称。

    返回:
    tuple: (combined_expression, combined_labels, label_mapping)
    """
    save_path = os.path.join(output_path, species_name, tissue_name)
    combined_expression = pd.read_csv(os.path.join(save_path, f"{species_name}_{tissue_name}_combined_expression.csv"),
                                      index_col=0)
    combined_labels = np.load(os.path.join(save_path, f"{species_name}_{tissue_name}_combined_labels.npy"))
    label_mapping = joblib.load(os.path.join(save_path, f"{species_name}_{tissue_name}_label_mapping.pkl"))

    return combined_expression, combined_labels, label_mapping


# def preprocess_special_file(file_path, label_mapping):
#     """
#     预处理包含特殊格式的基因表达文件。
#
#     参数:
#     file_path (str): 文件路径。
#     label_mapping (dict): 细胞类型到标签的映射字典。
#
#     返回:
#     tuple: (基因表达DataFrame, 标签列表)
#     """
#     df = pd.read_csv(file_path, sep='\t', header=None)
#     genes = df.iloc[:, 0].where(df.iloc[:, 0] != '--', df.iloc[:, 1])
#     df['gene_id'] = genes
#     df['expression_values'] = df.iloc[:, 4].apply(lambda x: np.array(x.split(', '), dtype=float))
#
#     # 删除原始列并设置新列名
#     df = df.drop(columns=[0, 1, 4])
#     df.columns = ['cell_type_id', 'cell_label', 'gene_id', 'expression_values']
#
#     df = df.drop_duplicates(subset=['gene_id', 'cell_type_id'])
#
#     gene_ids_ordered = list(OrderedDict.fromkeys(df['gene_id']))
#     expression_matrices = []
#     cell_labels = []
#
#     for cell_type_id, group in df.groupby('cell_type_id'):
#         expression_values = np.vstack(group['expression_values'].values)
#         expression_matrices.append(expression_values)
#         cell_label = group['cell_label'].iloc[0]
#         mapped_label = label_mapping.get(cell_label, len(label_mapping))
#         label_mapping[cell_label] = mapped_label
#         cell_labels.extend([mapped_label] * expression_values.shape[1])
#
#     expression = np.hstack(expression_matrices)
#
#     # 创建 AnnData 对象
#     adata = sc.AnnData(X=expression.T)
#     adata.var_names = gene_ids_ordered
#     adata.obs_names = [f"cell_{i}" for i in range(adata.shape[0])]
#
#     # 进行质量控制和过滤
#     adata = quality_control_and_filter(adata)
#
#     # 过滤后的细胞标签
#     filtered_cell_labels = [cell_labels[int(i.split('_')[1])] for i in adata.obs_names]
#
#     # 进行标准化
#     # scaler = MinMaxScaler()
#     # normalized_expression = scaler.fit_transform(expression)
#     # expression_df = pd.DataFrame(normalized_expression, index=gene_ids_ordered).sort_index()
#     normalized_expression = seurat_normalize(adata.X.T)
#     expression_df = pd.DataFrame(normalized_expression, columns=adata.obs_names, index=adata.var_names)
#
#     return expression_df, filtered_cell_labels


def preprocess_standard_files(gene_expression_file, cell_type_file, label_mapping):
    """
    预处理基因表达和细胞类型文件，判断细胞类型是细胞总型还是亚型并进行映射。

    参数:
    gene_expression_file (str): 基因表达文件路径。
    cell_type_file (str): 细胞类型文件路径。
    label_mapping (dict): 细胞类型到标签的映射字典。

    返回:
    tuple: (基因表达DataFrame, 标签列表)
    """
    gene_df = pd.read_csv(gene_expression_file, sep='\t', header=0)
    cell_type_df = pd.read_csv(cell_type_file, sep='\t', header=0)
    # 根据第一列（细胞名称）和 'Cell_type' 列进行过滤
    # 首先找到所有细胞类型为 'Uncharacterized' 的细胞名称
    uncharacterized_cells = cell_type_df[cell_type_df['Cell_type'] == 'Uncharacterized'].iloc[:, 0]

    # 从 gene_df 中删除这些细胞对应的列
    gene_df = gene_df.drop(columns=uncharacterized_cells, errors='ignore')

    # 同时从 cell_type_df 中删除这些细胞
    cell_type_df = cell_type_df[cell_type_df['Cell_type'] != 'Uncharacterized']

    # 加载统一的细胞类型映射文件
    subtype_to_type_mapping = load_cell_type_mapping('Celltype/Celltype.xlsx')

    # 转置基因表达矩阵
    gene_expression = gene_df.T
    gene_expression.columns = gene_expression.iloc[0]
    gene_expression = gene_expression.drop(gene_expression.index[0])

    # 对细胞类型进行处理，判断是总型还是亚型
    cell_labels = cell_type_df.set_index(cell_type_df.columns[0]).iloc[:, 0]

    # 将亚型转换为总型
    def map_to_major_type(cell_type):
        # 如果细胞类型存在于映射字典中，则为亚型，映射为总型
        if cell_type in subtype_to_type_mapping:
            return subtype_to_type_mapping[cell_type]
        # 否则，认为该类型为总型，直接返回
        return cell_type

    # 将细胞类型进行映射，优先判断是否为总型，否则转换为总型
    mapped_labels = cell_labels.map(lambda x: map_to_major_type(x))

    # 将映射后的总型转换为标签
    labels = mapped_labels.map(lambda x: label_mapping.setdefault(x, len(label_mapping))).values

    # 创建 AnnData 对象并进行标准化和过滤
    expression_matrix = gene_expression.values.astype(float)  # 确保数据类型为浮点数
    adata = sc.AnnData(X=expression_matrix)
    adata.var_names = gene_expression.columns
    adata.obs_names = gene_expression.index

    # 进行质量控制和标准化
    adata = quality_control_and_filter(adata)
    normalized_expression = seurat_normalize(adata.X.T)

    # 重新生成 DataFrame
    expression_df = pd.DataFrame(normalized_expression.T, columns=adata.var_names, index=adata.obs_names)
    expression_df = expression_df.T

    # 过滤后的标签
    filtered_labels = [labels[i] for i in range(len(labels)) if gene_expression.index[i] in adata.obs_names]

    return expression_df, filtered_labels



def align_gene_expression(predict_data, required_genes):
    """
    对齐基因表达矩阵的基因顺序。

    参数:
    predict_data (DataFrame): 基因表达矩阵。
    required_genes (list): 需要对齐的基因列表。

    返回:
    DataFrame: 对齐后的基因表达矩阵。
    """
    # 确保 required_genes 是列表格式
    required_genes = list(required_genes)

    # 获取 DataFrame 中现有的基因名，这些基因名是 DataFrame 的行索引
    existing_genes = set(predict_data.index)

    # 找出缺失的基因
    missing_genes = set(required_genes) - existing_genes
    if missing_genes:
        # 创建一个新的 DataFrame，包含所有缺失的基因，初始化为0
        missing_data = pd.DataFrame(0.0, index=list(missing_genes), columns=predict_data.columns)
        # 使用 pd.concat 一次性将所有缺失的行添加到原始 DataFrame 中
        predict_data = pd.concat([predict_data, missing_data])

    # 重新排序预测数据中的行，使其与训练数据的行顺序一致
    predict_data = predict_data.loc[required_genes]

    return predict_data


def prepare_graph_data(gene_expression, labels=None, k=5):
    """
    根据给定的基因表达矩阵和标签，生成图神经网络所需的数据对象。

    参数:
    gene_expression (np.array): 基因表达矩阵，细胞为行，基因为列。
    labels (np.array): 相应的细胞类型标签数组。
    k (int): 每个节点的最近邻数量。

    返回:
    Data: 用于图神经网络的数据对象。
    """
    # 确保gene_expression是DataFrame或类似的格式，然后转换为NumPy数组
    gene_expression = gene_expression.values if isinstance(gene_expression, pd.DataFrame) else gene_expression
    # 转换基因表达矩阵为Tensor，细胞为行，基因为列
    x = torch.tensor(gene_expression, dtype=torch.float)

    # 使用 sklearn 的 NearestNeighbors 找到 k 个最近邻
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(gene_expression)
    distances, indices = nbrs.kneighbors(gene_expression)

    # 构建 edge_index
    edge_index = []
    edge_attr = []
    for i in range(indices.shape[0]):
        for j in range(1, k + 1):  # 从1开始，跳过自身
            neighbor = indices[i][j]
            edge_index.append([i, neighbor])
            edge_attr.append(distances[i][j])  # 使用距离作为边属性

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 转换标签为Tensor
    y = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


def add_noise_to_features(features, noise_level=0.01):
    """
    向特征矩阵添加噪声。

    参数:
    features (Tensor): 特征矩阵。
    noise_level (float): 噪声级别。

    返回:
    Tensor: 添加噪声后的特征矩阵。
    """
    noise = torch.randn(features.size()) * noise_level
    noisy_features = features + noise
    return noisy_features


def perturb_graph_structure(edge_index, num_nodes, edge_attr, perturb_rate=0.05):
    """
    扰动图结构，随机删除和添加边。

    参数:
    edge_index (Tensor): 边索引矩阵。
    num_nodes (int): 节点数量。
    edge_attr (Tensor): 边属性矩阵。
    perturb_rate (float): 扰动比例。

    返回:
    tuple: (扰动后的边索引矩阵, 扰动后的边属性矩阵)
    """
    # 随机选择一部分边进行删除
    num_edges = edge_index.size(1)
    num_perturb = int(num_edges * perturb_rate)
    remove_indices = torch.randperm(num_edges)[:num_perturb]

    # 删除选择的边
    edge_index_perturbed = edge_index.clone()
    edge_index_perturbed = torch.cat([edge_index_perturbed[:, :remove_indices[0]],
                                      edge_index_perturbed[:, remove_indices[-1] + 1:]], dim=1)
    edge_attr_perturbed = edge_attr.clone()
    edge_attr_perturbed = torch.cat([edge_attr_perturbed[:remove_indices[0]],
                                     edge_attr_perturbed[remove_indices[-1] + 1:]], dim=0)

    # 随机添加新边
    add_edges = torch.randint(0, num_nodes, (2, num_perturb), dtype=torch.long)
    edge_index_perturbed = torch.cat([edge_index_perturbed, add_edges], dim=1)

    # 对新边生成默认属性，例如默认为1
    add_edge_attrs = torch.ones(num_perturb, dtype=torch.float)  # 假设新边属性为1
    edge_attr_perturbed = torch.cat([edge_attr_perturbed, add_edge_attrs], dim=0)

    # 确保 edge_index 的格式为 [2, num_edges]
    edge_index_perturbed = edge_index_perturbed.long().contiguous()

    return edge_index_perturbed, edge_attr_perturbed


def visualize_attention_weights(data, attn_weights, labels=None):
    # 创建一个有向图
    G = nx.DiGraph()

    # 添加节点
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        label = labels[i] if labels is not None else 'node'
        G.add_node(i, label=label)

    # 添加边和权重
    edge_index = attn_weights[0].cpu().numpy()
    edge_weights = attn_weights[1].cpu().numpy().flatten()
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0][i], edge_index[1][i]
        if src != tgt:  # 避免添加自环边
            weight = edge_weights[i]
            G.add_edge(src, tgt, weight=weight)

    # 定义节点位置
    pos = nx.spring_layout(G)

    # 绘制节点
    node_labels = nx.get_node_attributes(G, 'label')
    unique_labels = list(set(node_labels.values()))
    color_map = {label: plt.cm.get_cmap('autumn')(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
    node_colors = [color_map[node_labels[i]] for i in G.nodes()]

    # 绘制边
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 绘制图形
    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.5)  # 节点颜色更淡
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1,
                                   width=2, alpha=1)  # 边颜色更深
    nx.draw_networkx_labels(G, pos, labels={})  # 不显示节点标签值

    # 创建颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Attention Weights')

    plt.title('Attention Weights Visualization')
    plt.show()


def create_species_tissue_path(species, tissue):
    """
    创建用于存储模型的目录路径。

    参数:
    species (str): 物种名称。
    tissue (str): 组织名称。

    返回:
    str: 创建的目录路径。
    """
    path = os.path.join('model', species, tissue)
    os.makedirs(path, exist_ok=True)
    return path


class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)  # 增加第一层的维度
        self.conv2 = GCNConv(64, num_classes)  # 添加一个中间层
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = ELU()
        self.leaky_relu = nn.LeakyReLU(0.01)  # 使用LeakyReLU，小斜率为0.01

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=1)


class GNNWithAttention(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_heads=8, hidden_dim=8):
        super(GNNWithAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.5)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=0.5)
        self.dropout = Dropout(0.5)
        self.elu = ELU()
        self.residual = Linear(num_features, hidden_dim * num_heads)  # 用于残差连接

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        res = self.residual(x)  # 计算残差连接
        x, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.elu(x + res)  # 加上残差
        x = self.dropout(x)
        x, attn_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.elu(x)  # 额外的激活函数层
        x = self.dropout(x)
        return F.log_softmax(x, dim=1), attn_weights1, attn_weights2

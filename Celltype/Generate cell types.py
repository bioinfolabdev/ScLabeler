import re
import os
import pandas as pd

def is_exception(cell_name):
    """
    判断给定的细胞名称是否为例外情况（单个大写字母后跟 'cell' 或 'cells'）。

    参数:
    cell_name (str): 细胞名称字符串

    返回:
    bool: 如果是例外情况则返回 True，否则返回 False
    """
    # 匹配单个大写字母的单词加上 'cell' 或 'cells' 的模式
    pattern = r'^[A-Z] cells?$'
    return bool(re.match(pattern, cell_name))

def clean_cell_name(cell_name):
    """
    去除细胞名称中的 'cell' 或 'cells'，并去除括号中的内容，用于判断是否为亚型，但保留例外列表中的细胞类型。
    """

    # 去掉括号及其内容
    cell_name = re.sub(r'\s*\(.*?\)', '', cell_name)
    # 去除 'cell' 或 'cells'，去掉末尾多余的 's'
    return cell_name.replace('cells', '').replace('cell', '').strip().rstrip('s').lower()

def get_all_cell_types(data_folder):
    """
    获取文件夹下所有细胞类型，并将细胞总型和细胞亚型进行分类。

    参数:
    data_folder (str): 数据文件夹路径。

    返回:
    DataFrame: 细胞类型的DataFrame，包含总型和亚型。
    """
    all_cell_types = set()  # 使用 set 来避免重复项

    # 遍历文件夹下的物种文件夹
    for species_folder in os.listdir(data_folder):
        species_path = os.path.join(data_folder, species_folder)

        if os.path.isdir(species_path):
            # 遍历物种文件夹下的组织文件夹
            for tissue_folder in os.listdir(species_path):
                tissue_path = os.path.join(species_path, tissue_folder)

                if os.path.isdir(tissue_path):
                    # 查找 _Celltype.txt 后缀文件
                    for file in os.listdir(tissue_path):
                        if file.endswith('_Celltype.txt'):
                            file_path = os.path.join(tissue_path, file)

                            # 读取文件并提取 Cell_type 列
                            try:
                                celltype_df = pd.read_csv(file_path, sep='\t')
                                if 'Cell_type' in celltype_df.columns:
                                    cell_types = celltype_df['Cell_type'].unique()
                                    # 过滤掉 "Uncharacterized" 类型
                                    cell_types = {ct for ct in cell_types if ct.lower() != 'uncharacterized'}
                                    all_cell_types.update(cell_types)
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")

    # 转换为列表，方便后续处理
    all_cell_types = list(all_cell_types)

    # 创建一个DataFrame来存储结果
    df = pd.DataFrame({'Original_name': all_cell_types})

    # 将列中的值全部转换为字符串，避免 .str 方法出错
    df['Original_name'] = df['Original_name'].astype(str)

    # 按照 Original_name 的字符数量升序排序
    df = df.sort_values(by='Original_name', key=lambda x: x.str.len()).reset_index(drop=True)

    # 去除'cell'和'cells'后缀，去除括号内容，生成临时的处理列，保留例外情况
    df['Cleaned_name'] = df['Original_name'].apply(clean_cell_name)

    # 初始化两列
    df['Cell-type'] = df['Original_name']  # 默认原始名称是总型
    df['Cell-subtype'] = 'NA'  # 默认没有亚型，全部设为NA

    # 匹配并分类总型和亚型
    for i, general_type in df.iterrows():
        if df.at[i, 'Cell-subtype'] != 'NA':  # 跳过已经被标记为亚型的细胞类型
            continue

        general_name = general_type['Cleaned_name'].strip()

        for j, subtype in df.iterrows():
            if i != j and df.at[j, 'Cell-subtype'] == 'NA':  # 只处理未被标记为亚型的细胞类型
                subtype_name = subtype['Cleaned_name'].strip()

                if is_exception(general_type['Original_name']):
                    # 例外情况：匹配 T cells, B cells 等
                    if subtype_name.split()[-1] == general_name:
                        df.at[j, 'Cell-subtype'] = subtype['Original_name']
                        df.at[j, 'Cell-type'] = general_type['Original_name']
                        df.at[i, 'Cell-type'] = general_type['Original_name']
                elif subtype_name.endswith(general_name):
                    # 总型无括号，亚型有括号
                    if '(' not in general_type['Original_name'] and '(' in subtype['Original_name']:
                        df.at[j, 'Cell-subtype'] = subtype['Original_name']
                        df.at[j, 'Cell-type'] = general_type['Original_name']
                    elif '(' not in subtype['Original_name'] and len(general_name) <= len(subtype_name):
                        df.at[j, 'Cell-subtype'] = subtype['Original_name']
                        df.at[j, 'Cell-type'] = general_type['Original_name']

    # 去除临时处理列
    df = df.drop(columns=['Cleaned_name'])

    # 去除已经被标记为亚型的行
    df = df[df['Cell-type'] != 'NA']

    # 按 Cell-type 升序排序
    return df.sort_values(by='Cell-type').reset_index(drop=True)

# 使用示例
data_folder = '../data copy'  # 数据文件夹路径
cell_types_df = get_all_cell_types(data_folder)

# 设置输出文件夹为当前文件夹
output_folder = '.'
os.makedirs(output_folder, exist_ok=True)

# 将 DataFrame 保存到 xlsx 文件中
output_file = os.path.join(output_folder, 'Celltype.xlsx')
cell_types_df.to_excel(output_file, index=False)

print(f"结果保存到: {output_file}")

import os
import pandas as pd

# 定义根文件夹路径
root_folder_path = 'original data'

# 遍历所有物种文件夹
for species_folder in os.listdir(root_folder_path):
    species_folder_path = os.path.join(root_folder_path, species_folder)

    if os.path.isdir(species_folder_path):
        # 遍历物种文件夹内的所有组织文件夹
        for tissue_folder in os.listdir(species_folder_path):
            tissue_folder_path = os.path.join(species_folder_path, tissue_folder)

            if os.path.isdir(tissue_folder_path):
                # 获取文件列表
                files = os.listdir(tissue_folder_path)

                # 初始化字典来存储文件对
                file_pairs = {}

                # 遍历文件列表，根据文件名后缀进行分组
                for file_name in files:
                    if '_celltype' in file_name:
                        prefix = file_name.replace('_celltype.csv', '')
                        if prefix not in file_pairs:
                            file_pairs[prefix] = {}
                        file_pairs[prefix]['celltype'] = file_name
                    elif '_data' in file_name:
                        prefix = file_name.replace('_data.csv', '')
                        if prefix not in file_pairs:
                            file_pairs[prefix] = {}
                        file_pairs[prefix]['data'] = file_name
                
                # 遍历每对文件，执行处理逻辑
                for prefix, pair in file_pairs.items():
                    if 'data' in pair:
                        data_file = os.path.join(tissue_folder_path, pair['data'])

                        # 加载数据文件
                        data_df = pd.read_csv(data_file, header=0, index_col=0)

                        # 提取 species 和 tissue
                        species = species_folder  # 物种从文件夹名称获取
                        tissue = tissue_folder  # 组织从文件夹名称获取

                        # 如果存在 celltype 文件，则加载并处理 celltype 信息
                        if 'celltype' in pair:
                            celltype_file = os.path.join(tissue_folder_path, pair['celltype'])
                            celltype_df = pd.read_csv(celltype_file, header=0)
                            
                            # 处理与 celltype 相关的逻辑
                            celltype_df.set_index('Cell', inplace=True)
                            filtered_celltype_df = celltype_df.loc[celltype_df.index.isin(data_df.columns)]
                            filtered_celltype_df.reset_index(inplace=True)
                            filtered_celltype_df = filtered_celltype_df[['Cell', 'Cell_type']]

                            # 输出处理后的 celltype 信息（如果有）
                            print(f"Processed celltype file for {prefix}")
                        else:
                            # 如果没有 celltype 文件，可以设置一个默认值或跳过相关逻辑
                            filtered_celltype_df = pd.DataFrame(columns=['Cell', 'Cell_type'])
                            print(f"No celltype file for {prefix}, skipping celltype processing.")
                        
                        # 继续执行与 data 相关的处理逻辑
                        filtered_data_df = data_df.loc[:, (data_df != 0).any(axis=0)]
                        filtered_data_df = filtered_data_df[(filtered_data_df != 0).any(axis=1)]

                        # 基因转换和过滤等逻辑
                        combined_species_gene_data = pd.read_csv('geneinfo/gene_info.txt', sep='\t')
                        filtered_data_df.index = filtered_data_df.index.str.upper()
                        valid_genes = combined_species_gene_data.loc[
                            combined_species_gene_data['species'] == species.capitalize()
                            ].copy()
                        valid_genes.loc[:, 'Symbol'] = valid_genes['Symbol'].str.upper()
                        valid_genes.loc[:, 'Synonyms'] = valid_genes['Synonyms'].str.upper()

                        gene_to_symbol = {}
                        for gene in filtered_data_df.index:
                            gene = str(gene)  # 确保 gene 是字符串类型
                            symbol_matches = valid_genes.loc[valid_genes['Symbol'] == gene, 'Symbol']
                            if not symbol_matches.empty:
                                gene_to_symbol[gene] = symbol_matches.iloc[0]

                        for gene in filtered_data_df.index.difference(gene_to_symbol.keys()):
                            # 确保 gene 是字符串类型，避免非字符串类型导致错误
                            if isinstance(gene, str):
                                synonym_matches = valid_genes.loc[
                                    valid_genes['Synonyms'].str.contains(gene, na=False), 'Symbol']
                            else:
                                synonym_matches = pd.Series([])  # 如果 gene 不是字符串，返回空匹配
                            if not synonym_matches.empty:
                                gene_to_symbol[gene] = synonym_matches.iloc[0]

                        symbol_counts = pd.Series(gene_to_symbol.values()).value_counts()
                        duplicated_symbols = symbol_counts[symbol_counts > 1].index

                        for gene, symbol in list(gene_to_symbol.items()):
                            if symbol in duplicated_symbols:
                                print(f"Removing duplicated gene {gene} with Symbol {symbol}")
                                del gene_to_symbol[gene]

                        final_genes = pd.Index(gene_to_symbol.keys())
                        filtered_data_df = filtered_data_df.loc[final_genes]

                        # 生成最终文件名并保存
                        num_cells = filtered_data_df.shape[1]
                        data_output_filename = f"{species}_{tissue}{num_cells}_Data.txt"
                        data_output_path = os.path.join(tissue_folder_path, data_output_filename)

                        # 保存处理后的文件
                        filtered_data_df.to_csv(data_output_path, sep='\t')

                        # 如果有 celltype 信息，保存对应的 celltype 文件
                        if not filtered_celltype_df.empty:
                            celltype_output_filename = f"{species}_{tissue}{num_cells}_Celltype.txt"
                            celltype_output_path = os.path.join(tissue_folder_path, celltype_output_filename)
                            filtered_celltype_df.to_csv(celltype_output_path, sep='\t', index=False)

                        print(f"Processed data file for {prefix}")

                    else:
                        print(f"Missing pair for prefix: {prefix} in {tissue_folder_path}")

# 加载必要的包
library(Seurat)
library(ggplot2)
library(dplyr)
library(scales) # 用于生成颜色映射

# 1. 读取基因表达矩阵
expression_matrix <- read.table(
  "test data/human/Liver/human_Liver5105_Data.txt",
  header = TRUE, row.names = 1
)

# 2. 细胞预测结果文件路径
sclabeler_file <- "results/human/Liver/human_Liver5101_predictions.csv"

# 读取预测结果
sclabeler_results <- read.csv(sclabeler_file)

# 提取共有细胞
common_cells <- intersect(colnames(expression_matrix), sclabeler_results[[1]])

# 筛选基因表达矩阵
expression_matrix_filtered <- expression_matrix[, common_cells]

# 3. 创建 Seurat 对象并运行一次分析流程 -----------------------------------
# 创建 Seurat 对象
seurat_object <- CreateSeuratObject(counts = expression_matrix_filtered)

# Seurat 标准分析流程
seurat_object <- NormalizeData(seurat_object)
seurat_object <- FindVariableFeatures(seurat_object)
seurat_object <- ScaleData(seurat_object)
seurat_object <- RunPCA(seurat_object)
seurat_object <- RunUMAP(seurat_object, dims = 1:10)

# 4. 检查是否存在真实细胞类型并绘图 ----------------------------------------
# 假设第 2 列是预测细胞类型，第 3 列（如果存在）是真实细胞类型
predicted_cell_types <- sclabeler_results %>%
  filter(sclabeler_results[[1]] %in% common_cells) %>%
  rename(Predicted_Cell_Type = 2)

# 将预测细胞类型映射到元数据
seurat_object$Predicted_Cell_Type <- predicted_cell_types$Predicted_Cell_Type[match(colnames(seurat_object), sclabeler_results[[1]])]

# 初始化统一的颜色映射
cell_type_union <- unique(seurat_object$Predicted_Cell_Type)

# 检查是否存在真实细胞类型
if (ncol(sclabeler_results) >= 3) {
  true_cell_types <- sclabeler_results %>%
    filter(sclabeler_results[[1]] %in% common_cells) %>%
    rename(True_Cell_Type = 3)
  
  # 将真实细胞类型映射到元数据
  seurat_object$True_Cell_Type <- true_cell_types$True_Cell_Type[match(colnames(seurat_object), sclabeler_results[[1]])]
  
  # 合并真实和预测的细胞类型以生成统一颜色映射
  cell_type_union <- unique(c(cell_type_union, seurat_object$True_Cell_Type))
}

# 为所有细胞类型生成统一的颜色映射
cell_type_colors <- hue_pal()(length(cell_type_union))  # 为每个细胞类型生成不同的颜色
names(cell_type_colors) <- cell_type_union  # 将颜色与细胞类型对应

# 绘制预测细胞类型 UMAP 图
p_predicted <- DimPlot(seurat_object, reduction = "umap", group.by = "Predicted_Cell_Type") +
  scale_color_manual(values = cell_type_colors) +  # 使用统一颜色
  ggtitle("Predicted Cell Types")

# 保存预测细胞类型 UMAP 图
ggsave("visualization/Predicted_Cell_Types.png", plot = p_predicted, width = 8, height = 6, dpi = 300)

# 如果存在真实细胞类型，则绘制真实细胞类型 UMAP 图
if (ncol(sclabeler_results) >= 3) {
  p_true <- DimPlot(seurat_object, reduction = "umap", group.by = "True_Cell_Type") +
    scale_color_manual(values = cell_type_colors) +  # 使用统一颜色
    ggtitle("True Cell Types")
  
  # 保存真实细胞类型 UMAP 图
  ggsave("visualization/True_Cell_Types.png", plot = p_true, width = 8, height = 6, dpi = 300)
}

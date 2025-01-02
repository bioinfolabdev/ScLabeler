# 加载必要的包
library(Seurat)
library(ggplot2)
library(dplyr)

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

# 绘制预测细胞类型 UMAP 图
p_predicted <- DimPlot(seurat_object, reduction = "umap", group.by = "Predicted_Cell_Type") +
  ggtitle("Predicted Cell Types")

# 保存预测细胞类型 UMAP 图
ggsave("Predicted_Cell_Types.png", plot = p_predicted, width = 8, height = 6, dpi = 300)

# 检查是否有真实细胞类型列（假设为第 3 列）
if (ncol(sclabeler_results) >= 3) {
  true_cell_types <- sclabeler_results %>%
    filter(sclabeler_results[[1]] %in% common_cells) %>%
    rename(True_Cell_Type = 3)
  
  # 将真实细胞类型映射到元数据
  seurat_object$True_Cell_Type <- true_cell_types$True_Cell_Type[match(colnames(seurat_object), sclabeler_results[[1]])]
  
  # 绘制真实细胞类型 UMAP 图
  p_true <- DimPlot(seurat_object, reduction = "umap", group.by = "True_Cell_Type") +
    ggtitle("True Cell Types")
  
  # 保存真实细胞类型 UMAP 图
  ggsave("True_Cell_Types.png", plot = p_true, width = 8, height = 6, dpi = 300)
}

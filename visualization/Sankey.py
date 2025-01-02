import pandas as pd
from pyecharts.charts import Sankey
from pyecharts import options as opts
import webbrowser

# 读取数据
file_path = '../results/human/Liver/human_Liver5101_predictions.csv'  # 替换为实际文件路径
data = pd.read_csv(file_path)

# 按 True Cell Types 和 Predicted Cell Types 统计流向和数量
flows = data.groupby(['True Cell Types', 'Predicted Cell Types']).size().reset_index(name='value')
flows_file_path = '细胞类别统计.csv'
flows.to_csv(flows_file_path, index=False)

# 增加上下文区分节点
flows['True Cell Types'] = flows['True Cell Types'] + " (真实类别)"
flows['Predicted Cell Types'] = flows['Predicted Cell Types'] + " (预测类别)"

# 构建节点和连接数据
unique_true = flows['True Cell Types'].unique()
unique_predicted = flows['Predicted Cell Types'].unique()

# 合并所有节点，确保每个节点唯一
unique_nodes = pd.concat([pd.Series(unique_true), pd.Series(unique_predicted)]).unique()
nodes = [{"name": node} for node in unique_nodes]

# 构建链接数据
links = [
    {
        "source": flows.loc[i, "True Cell Types"],
        "target": flows.loc[i, "Predicted Cell Types"],
        "value": int(flows.loc[i, "value"])
    }
    for i in flows.index
]

# 绘制 Sankey 图
sankey = (
    Sankey(init_opts=opts.InitOpts(width="1200px", height="800px", page_title="Sankey Diagram Centered"))
    .add(
        '',  # 图例名称
        nodes,  # 节点数据
        links,  # 连接数据
        linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color="source"),  # 线条样式
        label_opts=opts.LabelOpts(
            position="right",  # 标签位置
            font_size=18,      # 标签字体大小
            font_family="Arial"  # 标签字体
        ),
        node_gap=30,  # 节点间距
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="human_Liver5101", 
            title_textstyle_opts=opts.TextStyleOpts(font_size=24),  # 标题字体大小
            pos_left="center",  # 标题水平居中
            pos_top="0px"      # 标题与图像顶部间距调整
        )
    )
)

# 保存并自动打开
output_file = 'human_Liver5101_sankey.html'  # 输出文件路径
sankey.render(output_file)
print(f"Sankey 图已保存为 {output_file}")
webbrowser.open(output_file)

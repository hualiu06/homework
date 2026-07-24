"""
Hugging Face 模型数据分析脚本
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============ 全局配置 ============

# 输出目录：脚本所在文件夹
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 第1步：加载数据 ==========
# 从爬虫输出的合并 CSV 中读取 Hugging Face 模型数据集
print("="*60)
print("Loading data...")
print("="*60)
df = pd.read_csv(f"{OUTPUT_DIR}/hf_models_all.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ========== 第2步：数据清洗与标准化 ==========
# 处理缺失值、重复项、异常值，并对分类字段做统一映射
print("\n" + "="*60)
print("PART 2: Data Cleaning & Analysis")
print("="*60)

print("\n--- Missing Value Analysis ---")
# 统计各列的缺失值数量及百分比
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing'] > 0])

# 用默认值填充缺失字段，确保后续分析不会因空值中断
df['license'] = df['license'].fillna('Unknown')          # 许可证未知的标记为 Unknown
df['arxiv'] = df['arxiv'].fillna('')                     # 无论文的用空字符串填充
df['size_category'] = df['size_category'].fillna('Unknown')  # 规模未知的标记为 Unknown
df['library_name'] = df['library_name'].fillna('Unknown')    # 库名称未知的标记为 Unknown

# 将字符串日期解析为 pandas 时间戳（datetime 类型）
# errors='coerce' 表示无法解析的日期会被设为 NaT（Not a Time），避免报错
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['last_modified'] = pd.to_datetime(df['last_modified'], errors='coerce')

# 以 model_id 为唯一标识去重，确保每个模型只保留一条记录
before = len(df)
df = df.drop_duplicates(subset=['model_id'])
print(f"\nRemoved {before - len(df)} duplicates, {len(df)} rows remaining")

# 对各任务类别的下载量做基本统计量分析
# 通过均值、中位数、四分位数等识别潜在的异常值
for tag in ['text-generation', 'image-classification', 'text-to-image']:
    subset = df[df['pipeline_tag'] == tag]
    print(f"\n{tag} - Downloads stats:")
    print(f"  Mean: {subset['downloads'].mean():,.0f}, Median: {subset['downloads'].median():,.0f}")
    print(f"  Min: {subset['downloads'].min():,}, Max: {subset['downloads'].max():,}")
    print(f"  Q1: {subset['downloads'].quantile(0.25):,.0f}, Q3: {subset['downloads'].quantile(0.75):,.0f}")

# 将 Hugging Face API 返回的简写许可证代码统一映射为标准的可读名称
# 例如 "apache-2.0" → "Apache 2.0", "mit" → "MIT"
license_mapping = {
    'apache-2.0': 'Apache 2.0',
    'mit': 'MIT',
    'mit-license': 'MIT',
    'bsd-3-clause': 'BSD 3-Clause',
    'bsd-2-clause': 'BSD 2-Clause',
    'bsd': 'BSD',
    'gpl-3.0': 'GPL 3.0',
    'lgpl-3.0': 'LGPL 3.0',
    'lgpl-2.1': 'LGPL 2.1',
    'mozilla-2.0': 'MPL 2.0',
    'mpl-2.0': 'MPL 2.0',
    'cc-by-4.0': 'CC BY 4.0',
    'cc-by-nc-4.0': 'CC BY-NC 4.0',
    'cc-by-nc-sa-4.0': 'CC BY-NC-SA 4.0',
    'cc-by-sa-4.0': 'CC BY-SA 4.0',
    'openrail': 'OpenRAIL',
    'openrail++': 'OpenRAIL++',
    'bigscience-openrail-m': 'OpenRAIL-M',
    'bigcode-openrail-m': 'OpenRAIL-M',
    'bigscience-bloom-1.3': 'RAIL',
    'llama2': 'Llama 2',
    'llama3': 'Llama 3',
    'llama3.1': 'Llama 3.1',
    'llama3.2': 'Llama 3.2',
    'other': 'Other',
    'unknown': 'Unknown',
    'deepseek': 'DeepSeek',
    'gemma': 'Gemma',
    'fair': 'Fair',
    'creativeml-openrail-m': 'OpenRAIL-M',
    'odc-by': 'ODC-BY',
    'afl-3.0': 'AFL 3.0',
    'agpl-3.0': 'AGPL 3.0',
    'unlicense': 'Unlicense',
    'apache-1.0': 'Apache 1.0',
}
df['license_clean'] = df['license'].map(
    lambda x: next((v for k, v in license_mapping.items() if k == x.lower()), x)
)

print("\n--- License Distribution ---")
print(df['license_clean'].value_counts().head(15).to_string())

# 将规模等级设为有序的 Categorical 类型，便于后续排序和编码
# 顺序从 <100M 到 100B+，Unknown 排在最后
size_order = ['<100M', '100M-1B', '1B-5B', '5B-10B', '10B-20B', '20B-50B', '50B-100B', '100B+', 'Unknown']
df['size_category'] = pd.Categorical(df['size_category'], categories=size_order, ordered=True)

# ========== 第3步：相关性分析（下载量 vs 点赞数）==========
# 探索下载量与点赞数之间的关系，同时按任务类别分组展示
# 如果两者高度相关，说明点赞数可以作为模型质量的代理指标
print("\n--- Correlation: Downloads vs Likes ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
tasks = [('text-generation', 'Text Generation'),
         ('image-classification', 'Image Classification'),
         ('text-to-image', 'Text to Image')]

corr_results = {}
for i, (tag, tname) in enumerate(tasks):
    subset = df[df['pipeline_tag'] == tag].copy()

    # 对下载量和点赞数做对数变换（log10）
    # 原因：少数模型拥有绝大多数下载量
    # 对数变换可以使数据分布更接近正态，便于做相关性分析
    subset['log_downloads'] = np.log10(subset['downloads'].clip(lower=1))
    subset['log_likes'] = np.log10(subset['likes'].clip(lower=1))

    # 计算皮尔逊相关系数以衡量线性关系
    r, p = stats.pearsonr(subset['log_downloads'], subset['log_likes'])
    # 计算斯皮尔曼等级相关系数
    rho, rho_p = stats.spearmanr(subset['downloads'], subset['likes'])
    corr_results[tag] = {'pearson_r': r, 'pearson_p': p,
                         'spearman_rho': rho, 'spearman_p': rho_p}

    print(f"\n  {tname}:")
    print(f"    Pearson r = {r:.4f}, p = {p:.6f}")
    print(f"    Spearman rho = {rho:.4f}, p = {rho_p:.6f}")

    # 绘制散点图：每个点代表一个模型
    ax = axes[i]
    ax.scatter(subset['log_downloads'], subset['log_likes'], alpha=0.6, s=30)

    # 添加线性回归拟合线，直观展示正/负相关趋势
    m, b = np.polyfit(subset['log_downloads'], subset['log_likes'], 1)
    x_line = np.linspace(subset['log_downloads'].min(), subset['log_downloads'].max(), 100)
    ax.plot(x_line, m * x_line + b, 'r--', alpha=0.8)

    ax.set_xlabel('Log10(Downloads)')
    ax.set_ylabel('Log10(Likes)')
    ax.set_title(f'{tname}\nPearson r={r:.3f}, Spearman ρ={rho:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_downloads_vs_likes.png", dpi=150)
plt.close()
print("\n  [Saved: correlation_downloads_vs_likes.png]")

# ========== 第4步：框架使用频率分析 ==========
# 统计每个任务类别中模型所使用的深度学习框架分布，这可以帮助了解社区的技术栈偏好
print("\n--- Framework Frequency Analysis ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (tag, tname) in enumerate(tasks):
    subset = df[df['pipeline_tag'] == tag]

    # 将逗号分隔的框架字段拆开统计
    # 一个模型可能支持多个框架（如 transformers + pytorch + onnx）
    all_frameworks = []
    for fw_list in subset['frameworks']:
        if isinstance(fw_list, str):
            all_frameworks.extend([fw.strip() for fw in fw_list.split(',')])

    # 统计各框架出现频次，取 Top 8 做可视化
    fw_counts = pd.Series(all_frameworks).value_counts()
    top_fw = fw_counts.head(8)

    print(f"\n  {tname} - Top Frameworks:")
    for fw, cnt in top_fw.items():
        pct = cnt / len(subset) * 100
        print(f"    {fw}: {cnt} models ({pct:.1f}%)")

    # 绘制横向柱状图，每个条对应一个框架
    ax = axes[i]
    colors = plt.cm.Paired(np.linspace(0, 1, len(top_fw)))
    bars = ax.barh(range(len(top_fw)), top_fw.values, color=colors)
    ax.set_yticks(range(len(top_fw)))
    ax.set_yticklabels(top_fw.index)
    ax.set_xlabel('Number of Models')
    ax.set_title(f'{tname} - Framework Distribution')

    # 在柱状图末端标注具体数值
    for bar, val in zip(bars, top_fw.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/framework_distribution.png", dpi=150)
plt.close()
print("\n  [Saved: framework_distribution.png]")

# ========== 第5步：机器学习建模 — 文本生成模型流行度归因 ==========
# 使用随机森林回归模型，分析哪些因素最能预测模型的下载量
print("\n" + "="*60)
print("PART 3: ML Model - What Drives Popularity? (Text Generation)")
print("="*60)

# 选取文本生成任务进行建模
tg = df[df['pipeline_tag'] == 'text-generation'].copy()
print(f"Text Generation models for modeling: {len(tg)}")

# ========== 特征工程 ==========
# 将原始字段转换为机器学习模型可用的数值特征

# 二值特征：是否有学术论文（0/1）
tg['has_arxiv'] = tg['arxiv'].astype(bool).astype(int)

# 二值特征：是否已知参数量（0/1）
tg['has_params'] = tg['param_count'].notna().astype(int)

# 将参数量从原始数值转为以十亿为单位，便于模型理解数值尺度
tg['param_count_billions'] = tg['param_count'] / 1e9

# 对目标变量做以 10 为底的对数变换，原因：下载量分布极为不均（头部模型数千万，尾部只有个位数）
# 对数变换可以压缩数值范围，使模型更好地学习模式
tg['log_downloads'] = np.log10(tg['downloads'].clip(lower=1))
tg['log_likes'] = np.log10(tg['likes'].clip(lower=1))

# 对分类特征做数值编码：许可证名称
# LabelEncoder 将每个类别映射为一个整数（如 Apache 2.0 → 0, MIT → 1, ...）
tg['license_encoded'] = LabelEncoder().fit_transform(tg['license_clean'].fillna('Unknown'))

# 对分类特征做数值编码：主要框架（取框架列表中的第一项作为主框架）
tg['primary_fw'] = tg['frameworks'].apply(
    lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown'
)
tg['fw_encoded'] = LabelEncoder().fit_transform(tg['primary_fw'])

# 对有序分类特征做数值映射：参数量规模等级 → 数字
# 这里是有序映射（<100M=1, 100M-1B=2, ..., 100B+=8），保留了大小顺序信息
size_map = {'<100M': 1, '100M-1B': 2, '1B-5B': 3, '5B-10B': 4,
            '10B-20B': 5, '20B-50B': 6, '50B-100B': 7, '100B+': 8, 'Unknown': 0}
tg['size_encoded'] = tg['size_category'].map(size_map).fillna(0).astype(int)

# 计算模型距今发布的天数，用于衡量"先发优势"
# 去除时区信息（tz_localize(None)），使时间运算兼容
tg['created_at'] = tg['created_at'].dt.tz_localize(None)
tg['days_since_creation'] = (pd.Timestamp.now() - tg['created_at']).dt.days.fillna(0)

# 定义用于预测下载量的特征列表
feature_sets = {
    'downloads': ['license_encoded', 'fw_encoded', 'size_encoded',
                  'param_count_billions', 'days_since_creation', 'has_arxiv']
}

# ========== 模型训练与评估 ==========
# 丢弃含缺失值的样本，保证训练集完整性
for target_name, features in feature_sets.items():
    print(f"\n--- Predicting {target_name.upper()} ---")
    y_name = 'log_downloads' if target_name == 'downloads' else 'log_likes'
    model_df = tg[features + [y_name]].dropna()
    X = model_df[features]
    y = model_df[y_name]

    print(f"  Samples after dropping NA: {len(X)}")
    print(f"  Features: {features}")

    # 按 80%/20% 划分训练集和测试集
    # random_state=42 保证每次划分结果一致，便于复现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 随机森林回归器配置：
    #   n_estimators=200: 200 棵决策树，提高预测稳定性
    #   max_depth=8: 限制单棵树最大深度，防止过拟合
    #   min_samples_leaf=3: 叶节点最少样本数，进一步防止过拟合
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        random_state=42, min_samples_leaf=3
    )
    rf.fit(X_train, y_train)

    # 在测试集上评估模型性能
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)            # R²：模型解释了目标变量多少方差
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE：预测误差的均方根

    # 5 折交叉验证：将数据分成 5 份，轮流用 4 份训练 1 份验证
    # 这样可以更可靠地评估模型在未见数据上的表现
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

    print(f"  R^2 (test): {r2:.4f}")
    print(f"  RMSE (test): {rmse:.4f}")
    print(f"  CV R^2 (mean+/-std): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 输出特征重要性排序，找出哪些因素对预测下载量贡献最大
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n  Feature Importance ({target_name}):")
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

# 保存训练好的模型和特征名，供后续可视化复用
trained_rf = rf
trained_features = features

# ========== 特征重要性可视化 ==========
# 将随机森林的特征重要性绘制为横向柱状图，直观展示各因素的预测力
print("\n\n--- Feature Importance Visualization ---")

importance = pd.DataFrame({
    'feature': trained_features,
    'importance': trained_rf.feature_importances_
}).sort_values('importance', ascending=False)

# 将特征名映射为业务可读的中文标签
feature_labels = {
    'license_encoded': '开源协议',
    'fw_encoded': '框架',
    'size_encoded': '参数量级',
    'param_count_billions': '参数量',
    'days_since_creation': '模型年龄',
    'has_arxiv': '有学术论文'
}
importance['feature_label'] = importance['feature'].map(feature_labels).fillna(importance['feature'])

# 绘制横向柱状图，颜色从绿色渐变到红色
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))[::-1]
bars = ax.barh(range(len(importance)), importance['importance'].values, color=colors)
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance['feature_label'].values)
ax.set_xlabel('重要性')
ax.set_title('决定一个模型更受欢迎的核心因素是什么？')

# 在柱状图末端标注具体数值
for bar, val in zip(bars, importance['importance'].values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
plt.close()
print("  [Saved: feature_importance.png]")

print("\n\nAll analysis complete! Output files:")
print(f"  {OUTPUT_DIR}/correlation_downloads_vs_likes.png")
print(f"  {OUTPUT_DIR}/framework_distribution.png")
print(f"  {OUTPUT_DIR}/feature_importance.png")

"""
AI 岗位市场分析（中国城市级）：基于猎聘网爬取的真实招聘数据
涵盖岗位类型、城市分布、薪资水平、学历经验要求、行业分布和技能需求

数据来源: 猎聘网 (liepin.com) 爬虫采集
"""

import os, re, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无 GUI 环境报错
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置中文字体（优先 SimHei，回退到微软雅黑，最后用 DejaVu Sans 保底）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = plt.cm.Paired(np.linspace(0, 1, 20))
CITY_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# ========== 1. 数据加载 ==========
print("=" * 60)
print("1. 数据加载")
print("=" * 60)

# 读取爬虫采集的 CSV 数据
csv_path = os.path.join(OUTPUT_DIR, 'liepin_ai_jobs.csv')
if not os.path.exists(csv_path):
    print(f"[错误] 未找到爬取数据: {csv_path}")
    print("请先运行爬虫: python liepin_crawler.py")
    exit(1)

df = pd.read_csv(csv_path, encoding='utf-8-sig')
print(f"原始数据: {len(df)} 条")

# ========== 2. 数据清洗与特征工程 ==========
print("\n" + "=" * 60)
print("2. 数据清洗与特征工程")
print("=" * 60)

# ----- 缺失值填充 -----
df['city'] = df['city'].fillna('未知').astype(str)
df['education'] = df['education'].fillna('不限').astype(str)
df['company'] = df['company'].fillna('未知').astype(str)
df['industry'] = df['industry'].fillna('未知').astype(str)
df['skills'] = df['skills'].fillna('').astype(str)

# 清理城市名称中的方括号等符号
df['city'] = df['city'].str.replace(r'[\[\]【】]', '', regex=True).str.strip()

# ----- 薪资特征计算 -----
# 月薪取 (min + max) / 2 作为平均月薪
# 再乘以 12 得到年化薪资
df['salary_monthly_avg'] = df[['salary_min', 'salary_max']].mean(axis=1)
df['salary_annual'] = df['salary_monthly_avg'] * 12
# 对年薪取 log10，使目标变量更接近正态分布，有利于回归模型
df['salary_annual_log'] = np.log10(df['salary_annual'].clip(lower=10000))

# 剔除无薪资信息的记录（这些记录无法用于薪资分析）
before = len(df)
df = df.dropna(subset=['salary_min', 'salary_max'])
print(f"去除无薪资信息: {before - len(df)} 条，剩余 {len(df)} 条")

# ----- 学历标准化 -----
# 将各种学历表述统一为几个标准类别
edu_norm = {
    '博士': '博士', '硕士': '硕士', '研究生': '硕士',
    '本科': '本科', '统招本科': '本科',
    '大专': '大专', '不限': '不限', '其他': '其他',
}
df['education_cls'] = df['education'].map(lambda x: next((v for k, v in edu_norm.items() if k in x), '不限'))

# ----- 经验等级分类 -----
# 将经验年限映射到离散的等级区间
def map_exp(row):
    mn, mx = row['exp_min'], row['exp_max']
    if pd.isna(mn) or pd.isna(mx):
        return '经验不限'
    if mn == 0 and mx == 0:
        return '应届/在校'
    if mn == 0 and mx == 20:
        return '经验不限'
    avg = (mn + mx) / 2
    if avg <= 1:
        return '1年以下'
    elif avg <= 3:
        return '1-3年'
    elif avg <= 5:
        return '3-5年'
    elif avg <= 10:
        return '5-10年'
    else:
        return '10年以上'

df['exp_cls'] = df.apply(map_exp, axis=1)
EXP_ORDER = ['经验不限', '应届/在校', '1年以下', '1-3年', '3-5年', '5-10年', '10年以上']

# ----- 衍生特征 -----
# 是否为高级岗位（5 年以上经验）
df['is_senior'] = df['exp_cls'].apply(lambda x: 1 if x in ('5-10年', '10年以上') else 0)

# 年薪等级划分
df['salary_tier'] = pd.cut(df['salary_annual'],
    bins=[0, 120000, 240000, 360000, 480000, 720000, 99999999],
    labels=['<12万', '12-24万', '24-36万', '36-48万', '48-72万', '>72万'])

# ----- 技能特征提取 -----
# 合并原始 skills 字段中的技能
def combine_skills(row):
    sk = set()
    if row.get('skills'):
        for s in row['skills'].split(';'):
            if s:
                sk.add(s.strip())
    return ';'.join(sorted(sk))

df['skills_all'] = df.apply(combine_skills, axis=1)
df['skill_count'] = df['skills_all'].apply(lambda x: len(x.split(';')) if x else 0)

# 为每种技能创建 0/1 标志特征（用于后续建模和分析）
skill_patterns = {
    'has_python': r'python', 'has_pytorch': r'pytorch', 'has_tensorflow': r'tensorflow',
    'has_nlp': r'nlp|自然语言|bert|transformer',
    'has_cv': r'计算机视觉|cv|computer vision|opencv|image|vision',
    'has_llm': r'llm|大模型|gpt|langchain|rag',
    'has_aigc': r'aigc',
    'has_rec': r'推荐|rec',
    'has_ml': r'机器学习|深度学习|machine learning|deep learning',
    'has_cloud': r'云|cloud|aws|azure|gcp',
    'has_speech': r'语音|speech|audio',
}
for col, pat in skill_patterns.items():
    df[col] = df.apply(
        lambda row, p=pat: 1 if re.search(p, row['job_title'] + ' ' + row['skills_all'], re.I) else 0,
        axis=1
    )

# 输出清洗后的数据概览
print(f"\n清洗完成:")
print(f"  薪资范围: ¥{df['salary_annual'].min():,.0f} - ¥{df['salary_annual'].max():,.0f}/年")
print(f"  月薪中位数: ¥{df['salary_monthly_avg'].median():,.0f}")
print(f"  年薪中位数: ¥{df['salary_annual'].median():,.0f}")
print(f"  城市数: {df['city'].nunique()}")
print(f"  学历分布: {df['education_cls'].value_counts().to_dict()}")

# ========== 3. 探索性数据分析 ==========
print("\n" + "=" * 60)
print("3. 探索性数据分析")
print("=" * 60)

# 3.1 城市分布
print("\n--- 3.1 城市分布 ---")
city_counts = df['city'].value_counts().head(15)
print(city_counts.to_string())

# 3.2 岗位类别分布
print("\n--- 3.2 AI岗位类别分布 (Top 15) ---")
df['title_short'] = df['job_title'].str.replace(r'\(.*?\)', '', regex=True).str.strip().str[:40]
role_counts = df['title_short'].value_counts().head(15)
print(role_counts.to_string())

# 3.3 各城市薪资中位数
print("\n--- 3.3 各城市薪资中位数 (Top 12) ---")
city_sal = df.groupby('city')['salary_monthly_avg'].agg(['median', 'mean', 'count'])
city_sal = city_sal[city_sal['count'] >= 3].sort_values('median', ascending=False)
city_sal['median'] = city_sal['median'].apply(lambda x: f'¥{x:,.0f}')
city_sal['mean'] = city_sal['mean'].apply(lambda x: f'¥{x:,.0f}')
print(city_sal.head(12).to_string())

# 3.4 各经验等级薪资
print("\n--- 3.4 各经验等级薪资 ---")
exp_sal = df.groupby('exp_cls', observed=True)['salary_monthly_avg'].describe()
exp_sal = exp_sal.reindex([e for e in EXP_ORDER if e in exp_sal.index])
print(exp_sal.to_string())

# 3.5 学历分布
print("\n--- 3.5 学历分布 ---")
edu_counts = df['education_cls'].value_counts()
print(edu_counts.to_string())

print("\n--- 3.6 各学历薪资对比 ---")
edu_sal = df.groupby('education_cls', observed=True)['salary_monthly_avg'].describe()
print(edu_sal.to_string())

# 3.7 经验等级分布
print("\n--- 3.7 经验等级分布 ---")
exp_counts = df['exp_cls'].value_counts()
exp_counts = exp_counts.reindex([e for e in EXP_ORDER if e in exp_counts.index])
print(exp_counts.to_string())

# 3.8 行业分布
print("\n--- 3.8 行业分布 (Top 10) ---")
ind_counts = df['industry'].value_counts().head(10)
print(ind_counts.to_string())

# ----- 探索性分析可视化：2×3 子图布局 -----
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle(f'中国 AI 岗位市场分析 (猎聘网 {len(df)} 条真实招聘数据)', fontsize=16, fontweight='bold', y=1.01)

# 图1: AI 岗位城市分布柱状图
city_counts.head(12).plot(kind='bar', ax=axes[0, 0], color=COLORS[:12])
axes[0, 0].set_title('AI 岗位城市分布 (Top 12)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('岗位数量')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(city_counts.head(12).values):
    axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontsize=8)

# 图2: 各城市月薪中位数水平条形图
city_med = df.groupby('city')['salary_monthly_avg'].median().sort_values(ascending=False)
city_med = city_med[df.groupby('city')['salary_monthly_avg'].count() >= 2].head(12)
bars = axes[0, 1].barh(range(len(city_med)), city_med.values, color='orange')
axes[0, 1].set_yticks(range(len(city_med)))
axes[0, 1].set_yticklabels(city_med.index, fontsize=9)
axes[0, 1].set_title('各城市月薪中位数 (Top 12)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('月薪 (CNY)')
axes[0, 1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/10000:.1f}万'))
for bar, val in zip(bars, city_med.values):
    axes[0, 1].text(bar.get_width() + 2000, bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f}元', va='center', fontsize=8)
axes[0, 1].invert_yaxis()

# 图3: 经验等级分布柱状图
exp_counts = df['exp_cls'].value_counts()
exp_counts = exp_counts.reindex([e for e in EXP_ORDER if e in exp_counts.index]).dropna()
exp_counts.plot(kind='bar', ax=axes[0, 2], color='lightblue')
axes[0, 2].set_title('经验等级分布', fontsize=13, fontweight='bold')
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel('岗位数量')
axes[0, 2].tick_params(axis='x', rotation=15)

# 图4: 薪资 vs 经验等级柱状图
exp_med = df.groupby('exp_cls', observed=True)['salary_monthly_avg'].median()
exp_med = exp_med.reindex([e for e in EXP_ORDER if e in exp_med.index]).dropna()
exp_med.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('薪资中位数 vs 经验等级', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('经验等级')
axes[1, 0].set_ylabel('月薪 (CNY)')
axes[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/10000:.1f}万'))
axes[1, 0].tick_params(axis='x', rotation=15)
for i, v in enumerate(exp_med.values):
    axes[1, 0].text(i, v + 1000, f'{v:,.0f}元', ha='center', fontsize=8)

# 图5: 学历要求分布柱状图
edu_counts = df['education_cls'].value_counts()
edu_plot = edu_counts[edu_counts > 0]
edu_plot.plot(kind='bar', ax=axes[1, 1], color='salmon')
axes[1, 1].set_title('学历要求分布', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('岗位数量')
axes[1, 1].tick_params(axis='x', rotation=0)
for i, v in enumerate(edu_plot.values):
    axes[1, 1].text(i, v + 1, str(v), ha='center', fontsize=10)

# 图6: 行业分布柱状图
ind_counts.head(10).plot(kind='bar', ax=axes[1, 2], color='purple')
axes[1, 2].set_title('行业分布 (Top 10)', fontsize=13, fontweight='bold')
axes[1, 2].set_xlabel('')
axes[1, 2].set_ylabel('岗位数量')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'job_eda_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  [已保存: job_eda_overview.png（探索性分析总览）]")

# ========== 4. 技能需求分析 ==========
print("\n" + "=" * 60)
print("4. 技能需求分析")
print("=" * 60)

# 统计每种技能的出现频次（通过岗位名称中的关键词匹配）
skill_cats = ['has_python', 'has_pytorch', 'has_tensorflow', 'has_nlp', 'has_cv',
              'has_llm', 'has_aigc', 'has_rec', 'has_ml', 'has_cloud', 'has_speech']
skill_names = ['Python', 'PyTorch', 'TensorFlow', 'NLP/BERT', '计算机视觉',
               'LLM/大模型', 'AIGC', '推荐系统', 'ML/DL', '云平台', '语音']
skill_counts_series = pd.Series({name: df[col].sum() for col, name in zip(skill_cats, skill_names)})
skill_counts_series = skill_counts_series.sort_values(ascending=False)

print(f"\n--- 技能需求分布 (从岗位名称提取) ---")
for sk, cnt in skill_counts_series.items():
    print(f"  {sk}: {cnt} ({cnt/len(df)*100:.1f}%)")

# 按经验等级统计热门技能
print(f"\n--- 各经验等级 Top 技能 ---")
for exp in ['经验不限', '1-3年', '3-5年', '5-10年']:
    sub = df[df['exp_cls'] == exp]
    if len(sub) < 5:
        continue
    sub_skills = {name: sub[col].sum() for col, name in zip(skill_cats, skill_names)}
    top = sorted(sub_skills.items(), key=lambda x: -x[1])[:5]
    print(f"  [{exp}] {', '.join(f'{s}({c})' for s, c in top if c > 0)}")

# ----- 技能分析可视化：1×2 子图 -----
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左图: 技能需求柱状图
top_skills = skill_counts_series.head(12)
top_skills.plot(kind='bar', ax=axes[0], color=plt.cm.Set3(np.linspace(0, 1, len(top_skills))))
axes[0].set_title('AI 岗位技能需求 (从岗位名称提取)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('岗位数')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(top_skills.values):
    axes[0].text(i, v + 0.5, str(v), ha='center', fontsize=9)

# 右图: 技能对薪资的影响（有该技能 vs 无该技能的月薪中位数差值）
skill_impact = {}
for col, name in zip(skill_cats, skill_names):
    with_skill = df[df[col] == 1]['salary_monthly_avg'].median()
    without = df[df[col] == 0]['salary_monthly_avg'].median()
    if not pd.isna(with_skill) and not pd.isna(without):
        skill_impact[name] = with_skill - without
impact_s = pd.Series(skill_impact).sort_values(ascending=True)
impact_s.plot(kind='barh', ax=axes[1], color='teal')
axes[1].set_title('技能对月薪的影响 (有技能 - 无技能)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('月薪溢价 (CNY)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}元'))
for i, v in enumerate(impact_s.values):
    axes[1].text(v + 500 if v > 0 else v - 500, i, f'{v:,.0f}元', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'skill_demand_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  [已保存: skill_demand_analysis.png（技能分析）]")

# ========== 5. 薪资预测模型（随机森林） ==========
print("\n" + "=" * 60)
print("5. 薪资预测模型（随机森林）")
print("=" * 60)

df_model = df.dropna(subset=['salary_annual']).copy()

# 对类别特征进行数值编码（标签编码）
le_city = LabelEncoder()
le_exp = LabelEncoder()
le_edu = LabelEncoder()
le_industry = LabelEncoder()

df_model['city_enc'] = le_city.fit_transform(df_model['city'])
df_model['exp_enc'] = le_exp.fit_transform(df_model['exp_cls'])
df_model['edu_enc'] = le_edu.fit_transform(df_model['education_cls'])
df_model['industry_enc'] = le_industry.fit_transform(df_model['industry'].fillna('未知'))

# 模型特征列表：包括编码后的类别特征 + 数值特征 + 技能标志
model_features = ['city_enc', 'exp_enc', 'edu_enc', 'industry_enc',
                  'skill_count', 'is_senior'] + skill_cats

X = df_model[model_features]
y = df_model['salary_annual_log']  # 预测目标：对数变换后的年薪

# 划分训练集和测试集（80%/20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

# 随机森林回归模型
# 参数说明：200 棵树，最大深度 12，叶节点最少 5 个样本
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42,
                           min_samples_leaf=5, n_jobs=-1)
rf.fit(X_train, y_train)

# ----- 模型评估 -----
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
# 将对数预测值还原为实际薪资
y_test_actual = 10 ** y_test
y_pred_actual = 10 ** y_pred

print(f"\n--- 模型表现 ---")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f} (log10)")
print(f"MAE: {mae:.4f} (log10)")
print(f"实际平均: ¥{y_test_actual.mean():,.0f}/年")
print(f"预测平均: ¥{y_pred_actual.mean():,.0f}/年")
print(f"MAPE: {(abs(y_test_actual - y_pred_actual) / y_test_actual * 100).mean():.1f}%")

# 5 折交叉验证（Cross Validation），评估模型稳定性
cv = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"\n5折CV R2: {cv.mean():.4f} +/- {cv.std():.4f}")

# ----- 特征重要性 -----
importance = pd.DataFrame({
    'feature': model_features, 'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n--- 特征重要性 ---")
for _, r in importance.iterrows():
    print(f"  {r['feature']}: {r['importance']:.4f}")

# 特征中文标签映射
ft_labels = {
    'city_enc': '城市', 'exp_enc': '经验等级', 'edu_enc': '学历',
    'industry_enc': '行业', 'skill_count': '技能数量', 'is_senior': '高级岗位',
    'has_python': 'Python', 'has_pytorch': 'PyTorch', 'has_tensorflow': 'TensorFlow',
    'has_nlp': 'NLP/BERT', 'has_cv': '计算机视觉', 'has_llm': 'LLM/大模型',
    'has_aigc': 'AIGC', 'has_rec': '推荐系统', 'has_ml': 'ML/DL',
    'has_cloud': '云平台', 'has_speech': '语音',
}
importance['label'] = importance['feature'].map(ft_labels).fillna(importance['feature'])

# 特征重要性可视化（水平条形图，按重要性从高到低）
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))[::-1]
bars = ax.barh(range(len(importance)), importance['importance'].values, color=colors)
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance['label'].values, fontsize=11)
ax.set_xlabel('特征重要性', fontsize=13)
ax.set_title('AI 薪资驱动因素 (Random Forest 特征重要性)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, importance['importance'].values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'salary_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  [已保存: salary_feature_importance.png（特征重要性）]")

# ========== 6. 城市粒度分析 ==========
print("\n" + "=" * 60)
print("6. 城市粒度分析")
print("=" * 60)

# 6.1 各城市岗位数量、薪资、学历和高级岗位占比
print("\n--- 6.1 城市岗位数量 & 薪资 (Top 12) ---")
city_stats = df.groupby('city').agg(
    岗位数=('job_title', 'count'),
    月薪中位数=('salary_monthly_avg', 'median'),
    月薪均值=('salary_monthly_avg', 'mean'),
    学历硕士以上占比=('education_cls', lambda x: (x.isin(['硕士', '博士']).sum() / max(len(x), 1)) * 100),
    高级岗占比=('is_senior', lambda x: x.mean() * 100),
).sort_values('岗位数', ascending=False)
city_stats['月薪中位数'] = city_stats['月薪中位数'].apply(lambda x: f'¥{x:,.0f}')
city_stats['月薪均值'] = city_stats['月薪均值'].apply(lambda x: f'¥{x:,.0f}')
city_stats['学历硕士以上占比'] = city_stats['学历硕士以上占比'].apply(lambda x: f'{x:.0f}%')
city_stats['高级岗占比'] = city_stats['高级岗占比'].apply(lambda x: f'{x:.0f}%')
print(city_stats.head(12).to_string())

# 6.2 主要城市的学历要求构成
print("\n--- 6.2 主要城市学历要求构成 ---")
for city in ['北京', '上海', '深圳', '杭州', '成都', '广州']:
    sub = df[df['city'] == city]
    if len(sub) < 3:
        continue
    edu_pct = sub['education_cls'].value_counts(normalize=True).mul(100).round(0).astype(int)
    print(f"  {city} ({len(sub)} 条): {edu_pct.to_dict()}")

# 6.3 各城市 Top 3 技能需求
print("\n--- 6.3 主要城市技能需求 Top 3 ---")
for city in ['北京', '上海', '深圳', '杭州', '成都']:
    sub = df[df['city'] == city]
    if len(sub) < 3:
        continue
    city_skills = {name: sub[col].sum() for col, name in zip(skill_cats, skill_names)}
    top = sorted(city_skills.items(), key=lambda x: -x[1])[:3]
    print(f"  {city}: {', '.join(f'{s}({c})' for s, c in top if c > 0)}")

# ----- 城市分析可视化：1×2 子图 -----
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左图: Top 8 城市月薪箱线图
top8_cities = df['city'].value_counts().head(8).index
city_box_data = [df[df['city'] == c]['salary_monthly_avg'].dropna().values for c in top8_cities]
bp = axes[0].boxplot(city_box_data, patch_artist=True)
axes[0].set_xticklabels(top8_cities, fontsize=9)
for patch, color in zip(bp['boxes'], CITY_COLORS[:8]):
    patch.set_facecolor(color)
axes[0].set_title('各城市月薪分布 (Top 8)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('月薪 (CNY)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/10000:.1f}万'))
axes[0].tick_params(axis='x', rotation=45)

# 右图: 城市 × 学历热力图（显示各城市的学历要求占比）
city_edu = df[df['city'].isin(df['city'].value_counts().head(8).index)]
ct = pd.crosstab(city_edu['city'], city_edu['education_cls'], normalize='index') * 100
im = axes[1].imshow(ct.values, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(len(ct.columns)))
axes[1].set_xticklabels(ct.columns, fontsize=9)
axes[1].set_yticks(range(len(ct.index)))
axes[1].set_yticklabels(ct.index, fontsize=9)
for i in range(len(ct.index)):
    for j in range(len(ct.columns)):
        axes[1].text(j, i, f'{ct.values[i,j]:.0f}%', ha='center', va='center', fontsize=8)
axes[1].set_title('各城市学历要求占比', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1], shrink=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'city_analysis_detailed.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  [已保存: city_analysis_detailed.png（城市分析）]")

print(f"\n输出文件:")
print(f"  - job_eda_overview.png（探索性分析总览）")
print(f"  - skill_demand_analysis.png（技能分析）")
print(f"  - salary_feature_importance.png（特征重要性）")
print(f"  - city_analysis_detailed.png（城市粒度分析）")

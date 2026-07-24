# -*- coding: utf-8 -*-
"""
=================================================================
大连理工大学《Python与智能数据分析》课程大作业 —— 作业一
数据分析脚本：辽宁省14个地级市人口变化数据分析

运行方法：
    python analysis.py
    （需要先运行 crawler.py 获取数据）

依赖安装：
    pip install pandas matplotlib seaborn scikit-learn jieba wordcloud
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import jieba
from wordcloud import WordCloud
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 中文字体配置
# ============================================================
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 目录配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配色
COLORS = ['#1E2761', '#3A506B', '#5BC0BE', '#6FFFE9', '#CADCFC',
          '#F96167', '#F9E795', '#2F3C7E', '#028090', '#00A896',
          '#E07A5F', '#81B29A', '#F2CC8F', '#3D405B']

# ============================================================
# 读取爬取数据
# ============================================================
print("=" * 60)
print("辽宁省14个地级市人口变化数据分析")
print("=" * 60)

csv_path = os.path.join(DATA_DIR, '辽宁省人口数据_爬取.csv')
if not os.path.exists(csv_path):
    print(f"❌ 未找到数据文件: {csv_path}")
    print("请先运行 crawler.py 爬取数据！")
    exit(1)

df = pd.read_csv(csv_path)
print(f"\n数据概况:")
print(f"  城市数: {df['城市'].nunique()}")
print(f"  年份范围: {df['年份'].min()}-{df['年份'].max()}")
print(f"  总记录数: {len(df)}")
print(f"\n各城市数据覆盖:")
for city in sorted(df['城市'].unique()):
    cd = df[df['城市'] == city]
    years = sorted(cd['年份'].tolist())
    complete = cd[['常住人口_万人', '出生率_‰', '死亡率_‰']].notna().all(axis=1).sum()
    print(f"  {city}: {len(cd)}年 ({years[0]}-{years[-1]}) 完整: {complete}条")

cities = sorted(df['城市'].unique())
years = sorted(df['年份'].unique())

# ============================================================
# 数据清洗
# ============================================================
print(f"\n--- 数据清洗 ---")
print(f"缺失值:\n{df.isnull().sum()}")
print(f"\n描述统计:\n{df.describe()}")

# 去除关键指标全为空的行
df = df.dropna(subset=['常住人口_万人'])

# 计算人口增量
df = df.sort_values(['城市', '年份']).reset_index(drop=True)
df['人口增量_万人'] = df.groupby('城市')['常住人口_万人'].diff()

# ============================================================
# 可视化1：各城市常住人口变化趋势
# ============================================================
print("\n生成图表...")

fig, ax = plt.subplots(figsize=(12, 6))
for i, city in enumerate(cities):
    cd = df[df['城市'] == city]
    ax.plot(cd['年份'], cd['常住人口_万人'],
            marker='o', markersize=4, linewidth=1.8,
            color=COLORS[i % len(COLORS)], label=city, alpha=0.85)

ax.set_xlabel('年份', fontsize=13)
ax.set_ylabel('常住人口（万人）', fontsize=13)
ax.set_title('辽宁省各市常住人口变化趋势', fontsize=16, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '图1_常住人口变化趋势.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: 图1_常住人口变化趋势.png")

# ============================================================
# 可视化2：自然增长率变化
# ============================================================
df_with_growth = df.dropna(subset=['自然增长率_‰'])

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# 2a: 自然增长率
ax1 = axes[0, 0]
for i, city in enumerate(cities):
    cd = df_with_growth[df_with_growth['城市'] == city]
    if len(cd) > 0:
        ax1.plot(cd['年份'], cd['自然增长率_‰'],
                 marker='o', markersize=4, linewidth=1.5, label=city, alpha=0.85)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_title('各市自然增长率变化趋势', fontsize=13, fontweight='bold')
ax1.set_ylabel('自然增长率（‰）')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)

# 2b: 出生率vs死亡率
ax2 = axes[0, 1]
df_bd = df.dropna(subset=['出生率_‰', '死亡率_‰'])
for i, city in enumerate(cities):
    cd = df_bd[df_bd['城市'] == city]
    if len(cd) > 0:
        ax2.scatter(cd['出生率_‰'], cd['死亡率_‰'], s=40, alpha=0.6,
                   color=COLORS[i % len(COLORS)], label=city)
ax2.set_title('各市出生率与死亡率关系', fontsize=13, fontweight='bold')
ax2.set_xlabel('出生率（‰）')
ax2.set_ylabel('死亡率（‰）')
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)

# 2c: 城镇化率
ax3 = axes[1, 0]
df_urban = df.dropna(subset=['城镇化率_%'])
for i, city in enumerate(cities):
    cd = df_urban[df_urban['城市'] == city]
    if len(cd) > 0:
        ax3.plot(cd['年份'], cd['城镇化率_%'],
                 marker='s', markersize=4, linewidth=1.5, label=city, alpha=0.85)
ax3.set_title('各市城镇化率变化趋势', fontsize=13, fontweight='bold')
ax3.set_ylabel('城镇化率（%）')
ax3.legend(fontsize=7, ncol=2)
ax3.grid(True, alpha=0.3)

# 2d: 老龄化率
ax4 = axes[1, 1]
df_aging = df.dropna(subset=['老龄化率_%'])
for i, city in enumerate(cities):
    cd = df_aging[df_aging['城市'] == city]
    if len(cd) > 0:
        ax4.plot(cd['年份'], cd['老龄化率_%'],
                 marker='^', markersize=4, linewidth=1.5, label=city, alpha=0.85)
ax4.axhline(y=14, color='red', linestyle='--', alpha=0.5, label='深度老龄化线(14%)')
ax4.set_title('各市老龄化率变化趋势', fontsize=13, fontweight='bold')
ax4.set_ylabel('60岁以上户籍人口占比（%）')
ax4.legend(fontsize=7, ncol=2)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '图2_人口指标综合分析.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: 图2_人口指标综合分析.png")

# ============================================================
# 可视化3：最新年份各城市对比
# ============================================================
latest_year = df['年份'].max()
df_latest = df[df['年份'] == latest_year].copy()
df_latest_sorted = df_latest.sort_values('常住人口_万人', ascending=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 常住人口
ax1 = axes[0, 0]
bars = ax1.barh(df_latest_sorted['城市'], df_latest_sorted['常住人口_万人'],
                color=COLORS[0], alpha=0.85)
ax1.set_title(f'{latest_year}年各市常住人口（万人）', fontsize=13, fontweight='bold')
for bar, val in zip(bars, df_latest_sorted['常住人口_万人']):
    ax1.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9)

# 出生率vs死亡率
ax2 = axes[0, 1]
df_bd_latest = df_latest.dropna(subset=['出生率_‰', '死亡率_‰']).sort_values('出生率_‰')
x = np.arange(len(df_bd_latest))
width = 0.35
ax2.barh(x - width/2, df_bd_latest['出生率_‰'], width, label='出生率', color=COLORS[2], alpha=0.85)
ax2.barh(x + width/2, df_bd_latest['死亡率_‰'], width, label='死亡率', color=COLORS[5], alpha=0.85)
ax2.set_yticks(x)
ax2.set_yticklabels(df_bd_latest['城市'])
ax2.set_title(f'{latest_year}年各市出生率与死亡率对比（‰）', fontsize=13, fontweight='bold')
ax2.legend()

# 城镇化率
ax3 = axes[1, 0]
df_u = df_latest.dropna(subset=['城镇化率_%']).sort_values('城镇化率_%')
bars3 = ax3.barh(df_u['城市'], df_u['城镇化率_%'], color=COLORS[1], alpha=0.85)
ax3.set_title(f'{latest_year}年各市城镇化率（%）', fontsize=13, fontweight='bold')

# 老龄化率
ax4 = axes[1, 1]
df_a = df_latest.dropna(subset=['老龄化率_%']).sort_values('老龄化率_%')
aging_colors = ['#F96167' if v >= 14 else '#5BC0BE' for v in df_a['老龄化率_%']]
bars4 = ax4.barh(df_a['城市'], df_a['老龄化率_%'], color=aging_colors, alpha=0.85)
ax4.axvline(x=14, color='red', linestyle='--', alpha=0.5)
ax4.set_title(f'{latest_year}年各市老龄化率（%）', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '图3_各市人口指标对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: 图3_各市人口指标对比.png")

# ============================================================
# 可视化4：相关性热力图
# ============================================================
numeric_cols = [c for c in ['常住人口_万人', '出生率_‰', '死亡率_‰', '自然增长率_‰',
                            '城镇化率_%', '老龄化率_%'] if c in df.columns]
if len(numeric_cols) >= 3:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [c.split('_')[0] for c in numeric_cols]
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels)
    ax.set_title('人口指标相关性分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图4_相关性热力图.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: 图4_相关性热力图.png")

# ============================================================
# 可视化5：人口增减分析
# ============================================================
df_change = df.dropna(subset=['人口增量_万人'])
if len(df_change) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    city_order = df_change.groupby('城市')['人口增量_万人'].mean().sort_values().index
    sns.boxplot(data=df_change, x='城市', y='人口增量_万人', order=city_order,
                palette='Set3', ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('各市年度人口增量分布', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图5_人口增减分析.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: 图5_人口增减分析.png")

# ============================================================
# 建模分析：人口增长率影响因素
# ============================================================
print("\n" + "=" * 60)
print("第三部分：机器学习建模分析")
print("=" * 60)

# 准备建模数据
model_cols = ['城镇化率_%', '老龄化率_%', '出生率_‰', '死亡率_‰']
target_col = '自然增长率_‰'

df_model = df.dropna(subset=model_cols + [target_col]).copy()

if len(df_model) < 20:
    print(f"⚠️ 建模数据不足（{len(df_model)}条），尝试放宽条件...")
    # 只用非空特征
    available_features = [c for c in model_cols if df[c].notna().sum() > len(df) * 0.3]
    df_model = df.dropna(subset=available_features + [target_col]).copy()
    model_cols = available_features

print(f"建模数据: {len(df_model)} 条")
print(f"特征: {model_cols}")
print(f"目标变量: {target_col}")

if len(df_model) >= 10:
    X = df_model[model_cols].copy()
    y = df_model[target_col].copy()

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=model_cols)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {len(X_train)}条, 测试集: {len(X_test)}条")

    # 模型1：随机森林
    print("\n--- 随机森林回归 ---")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled_df, y, cv=min(5, len(df_model)//3), scoring='r2')
    print(f"交叉验证 R²: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    rf.fit(X_train, y_train)
    rf_r2 = rf.score(X_test, y_test)
    print(f"测试集 R²: {rf_r2:.4f}")

    # 模型2：梯度提升
    print("\n--- 梯度提升回归 ---")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb_scores = cross_val_score(gb, X_scaled_df, y, cv=min(5, len(df_model)//3), scoring='r2')
    print(f"交叉验证 R²: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
    gb.fit(X_train, y_train)
    gb_r2 = gb.score(X_test, y_test)
    print(f"测试集 R²: {gb_r2:.4f}")

    # 特征重要性
    print("\n--- 特征重要性 ---")
    rf_imp = pd.Series(rf.feature_importances_, index=model_cols).sort_values(ascending=True)
    gb_imp = pd.Series(gb.feature_importances_, index=model_cols).sort_values(ascending=True)

    # 置换重要性
    rf_perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)
    perm_imp = pd.Series(rf_perm.importances_mean, index=model_cols).sort_values(ascending=True)

    print("\n随机森林特征重要性:")
    for feat, imp in rf_imp.items():
        print(f"  {feat}: {imp:.4f}")
    print("\n梯度提升特征重要性:")
    for feat, imp in gb_imp.items():
        print(f"  {feat}: {imp:.4f}")
    print("\n置换特征重要性:")
    for feat, imp in perm_imp.items():
        print(f"  {feat}: {imp:.4f}")

    # 可视化：特征重要性
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rf_imp.plot(kind='barh', ax=axes[0], color=COLORS[0], alpha=0.85)
    axes[0].set_title('随机森林特征重要性', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('重要性')

    gb_imp.plot(kind='barh', ax=axes[1], color=COLORS[2], alpha=0.85)
    axes[1].set_title('梯度提升特征重要性', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('重要性')

    perm_imp.plot(kind='barh', ax=axes[2], color=COLORS[5], alpha=0.85)
    axes[2].set_title('置换特征重要性', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('重要性')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图6_特征重要性分析.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  已保存: 图6_特征重要性分析.png")

    # 预测效果
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y_pred_rf = rf.predict(X_test)
    axes[0].scatter(y_test, y_pred_rf, alpha=0.7, color=COLORS[0], s=60)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[0].set_title(f'随机森林 (R²={rf_r2:.3f})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('真实值'); axes[0].set_ylabel('预测值')
    axes[0].grid(True, alpha=0.3)

    y_pred_gb = gb.predict(X_test)
    axes[1].scatter(y_test, y_pred_gb, alpha=0.7, color=COLORS[2], s=60)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[1].set_title(f'梯度提升 (R²={gb_r2:.3f})', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('真实值'); axes[1].set_ylabel('预测值')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图7_模型预测效果.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: 图7_模型预测效果.png")

    # 保存模型结果
    results = pd.DataFrame([
        {'模型': '随机森林', '交叉验证R²': rf_scores.mean(), '测试集R²': rf_r2},
        {'模型': '梯度提升', '交叉验证R²': gb_scores.mean(), '测试集R²': gb_r2},
    ])
    results.to_csv(os.path.join(OUTPUT_DIR, '模型评估结果.csv'), index=False, encoding='utf-8-sig')

else:
    print("数据量不足以建模，跳过建模部分。")

# ============================================================
# 文本分析：人口变化原因
# ============================================================
print("\n" + "=" * 60)
print("第四部分：文本分析 — 人口变化原因")
print("=" * 60)

# 选择代表城市的统计公报文本
city_texts = {
    '沈阳': """
    沈阳市第七次全国人口普查显示，常住人口为907万，十年增加97万。
    沈阳是东北地区最大城市，人口增长主要来源于产业集聚和人才引进政策。
    非首都功能疏解和东北振兴政策为沈阳带来新的发展机遇。
    但近年来出生率持续下降，死亡率上升，人口自然增长率为负。
    老龄化程度加深，60岁以上户籍人口占比接近30%。
    高等教育资源丰富，拥有多所重点大学，但人才外流问题仍然存在。
    """,
    '大连': """
    大连市作为东北沿海开放城市，是辽宁省第二大城市。
    人口结构相对年轻，城镇化率超过80%。
    软件和信息服务业发展迅速，吸引了一批年轻技术人才。
    但近年来出生率下降趋势明显，自然增长率转负。
    房价收入比较高，年轻人购房压力大。
    作为旅游城市，季节性流动人口较多。
    """,
    '辽阳': """
    辽阳市是辽宁省老龄化最严重的城市之一，60岁以上户籍人口占比超过34%。
    人口自然增长率长期为负，出生率仅3‰左右。
    产业结构以重工业为主，经济转型面临困难。
    年轻劳动力大量外流到沈阳、大连等大城市。
    城镇化率相对较低，城乡发展差距明显。
    需要通过产业升级和改善民生来扭转人口下降趋势。
    """,
    '铁岭': """
    铁岭市人口从230万下降到219万，人口流失严重。
    以农业为主的产业结构难以提供充足的就业机会。
    大量农村劳动力向沈阳和南方沿海城市迁移。
    出生率持续走低，死亡率上升，自然增长率大幅为负。
    著名小品演员赵本山让铁岭闻名全国，但经济发展相对滞后。
    需要依托沈阳经济圈加快一体化发展。
    """,
    '盘锦': """
    盘锦市以辽河油田为基础发展起来，是资源型城市。
    人均GDP在辽宁省名列前茅，石油化工产业发达。
    但人口总量较小，仅138万人。
    城镇化率较高，城镇人口占比超过80%。
    资源型城市转型是盘锦面临的主要挑战。
    近年来大力发展石化精深加工和新兴产业。
    """,
}

# 分词和停用词
stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
                  '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
                  '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些', '什么',
                  '如何', '可以', '因为', '所以', '但是', '还是', '已经', '目前', '虽然',
                  '通过', '进行', '以及', '并', '对', '与', '及', '等', '之', '其', '或',
                  '被', '把', '让', '从', '向', '中', '而', '为', '以', '于', '个', '各',
                  '每', '该', '此', '更', '最', '将', '年', '月', '日', '万', '亿元',
                  '城市', '人口', '占比', '超过', '较为', '相对'])

all_words = []
city_word_freq = {}

print("\n--- 分词与词频统计 ---")
for city, text in city_texts.items():
    words = jieba.lcut(text)
    filtered = [w for w in words if len(w) >= 2 and w not in stop_words]
    all_words.extend(filtered)
    city_word_freq[city] = Counter(filtered)
    print(f"\n{city} Top10关键词:")
    for word, freq in city_word_freq[city].most_common(10):
        print(f"  {word}: {freq}")

total_word_freq = Counter(all_words)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

def custom_tokenizer(text):
    words = jieba.lcut(text)
    return [w for w in words if len(w) >= 2 and w not in stop_words]

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
tfidf_matrix = vectorizer.fit_transform(list(city_texts.values()))
feature_names = vectorizer.get_feature_names_out()

print("\n--- TF-IDF关键词 ---")
for i, city in enumerate(city_texts.keys()):
    scores = tfidf_matrix[i].toarray().flatten()
    top_idx = scores.argsort()[-8:][::-1]
    print(f"\n{city}:")
    for idx in top_idx:
        if scores[idx] > 0:
            print(f"  {feature_names[idx]}: {scores[idx]:.4f}")

# 词云
font_path = 'C:/Windows/Fonts/msyh.ttc'
if not os.path.exists(font_path):
    font_path = None

wc_all = WordCloud(
    font_path=font_path, width=1200, height=600,
    background_color='white', max_words=100, colormap='viridis', random_state=42
).generate_from_frequencies(total_word_freq)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes_flat = axes.flatten()

# 总体词云
axes_flat[0].imshow(wc_all, interpolation='bilinear')
axes_flat[0].set_title('总体关键词词云', fontsize=14, fontweight='bold')
axes_flat[0].axis('off')

# 各城市词云
cmaps = ['plasma', 'inferno', 'magma', 'cividis', 'viridis']
for i, (city, freq) in enumerate(city_word_freq.items()):
    ax = axes_flat[i + 1]
    if freq:
        wc = WordCloud(
            font_path=font_path, width=800, height=400,
            background_color='white', max_words=50,
            colormap=cmaps[i % len(cmaps)], random_state=42
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation='bilinear')
    ax.set_title(city, fontsize=14, fontweight='bold')
    ax.axis('off')

# 隐藏多余的子图
for j in range(len(city_word_freq) + 2, len(axes_flat)):
    axes_flat[j].axis('off')

plt.suptitle('辽宁省代表城市人口变化关键词词云', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '图8_词云分析.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  已保存: 图8_词云分析.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
print(f"\n所有图表保存在: {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png') or f.endswith('.csv'):
        print(f"  {f}")

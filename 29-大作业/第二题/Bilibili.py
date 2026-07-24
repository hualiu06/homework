# -*- coding: utf-8 -*-
"""
Bilibili 排行榜数据分析系统
功能：
1. 爬取B站全站排行榜前50名视频及热门评论
2. 分析各分区视频数量占比与互动率
3. 多维度分析评论（情感倾向、意图分类、质量评估）
4. 扩展数据挖掘（创作者分析、时序分析等）
"""

import requests
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# ==========================================
# 字体配置优化
# ==========================================
def setup_chinese_font():
    """配置中文字体，解决绘图乱码问题"""
    import platform
    
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统字体
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong']
    elif system == 'Darwin':  # macOS
        font_names = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic']
    else:  # Linux
        font_names = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei']
    
    # 尝试所有字体
    for font in font_names:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            # 创建一个测试图来验证字体
            fig, ax = plt.subplots(figsize=(0.1, 0.1))
            ax.text(0.5, 0.5, '测试', fontsize=1)
            plt.close(fig)
            print(f"成功加载字体: {font}")
            return True
        except:
            continue
    
    # 如果所有字体都失败，使用默认设置并给出警告
    print("警告: 未找到合适的中文字体，图表可能无法正常显示中文")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 初始化字体
setup_chinese_font()

def _get_char_width(char):
    """获取字符的显示宽度，中文字符占2个宽度，英文字符占1个"""
    if '\u4e00' <= char <= '\u9fff':
        return 2
    return 1

def _get_string_width(s):
    """获取字符串的显示宽度"""
    return sum(_get_char_width(c) for c in str(s))

def print_table(df):
    """自定义表格打印函数，处理中文字符对齐"""
    if df.empty:
        print("数据为空")
        return
    
    headers = list(df.columns)
    data = df.values.tolist()
    
    col_widths = []
    for i, header in enumerate(headers):
        max_width = _get_string_width(header)
        for row in data:
            cell_width = _get_string_width(row[i])
            if cell_width > max_width:
                max_width = cell_width
        col_widths.append(max_width + 2)
    
    total_width = sum(col_widths) + len(col_widths) + 1
    print('+' + '-' * (total_width - 2) + '+')
    
    header_line = '|'
    for i, header in enumerate(headers):
        width = col_widths[i]
        header_width = _get_string_width(header)
        padding = width - header_width
        header_line += ' ' + header + ' ' * padding + '|'
    print(header_line)
    
    print('+' + '-' * (total_width - 2) + '+')
    
    for row in data:
        row_line = '|'
        for i, cell in enumerate(row):
            width = col_widths[i]
            cell_str = str(cell)
            cell_width = _get_string_width(cell_str)
            padding = width - cell_width
            row_line += ' ' + cell_str + ' ' * padding + '|'
        print(row_line)
    
    print('+' + '-' * (total_width - 2) + '+')

# ==========================================
# 第一部分：数据爬取模块
# ==========================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com",
}

def get_rank_list():
    """获取B站全站排行榜前50名"""
    url = "https://api.bilibili.com/x/web-interface/ranking/v2"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["code"] != 0:
            print(f"接口返回错误: {data['message']}")
            return None
        
        rank_list = data.get("data", {}).get("list", [])[:50]
        print(f"成功获取 {len(rank_list)} 条排行榜视频")
        return rank_list
        
    except Exception as e:
        print(f"请求失败: {e}")
        return None

def get_hot_comments(aid, limit=10):
    """
    获取视频的热门评论
    :param aid: 视频aid（数字）
    :param limit: 获取条数
    """
    url = f"https://api.bilibili.com/x/v2/reply/main?oid={aid}&type=1&mode=0&pn=1&ps={limit}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["code"] != 0:
            print(f"获取评论失败 (aid={aid}): {data['message']}")
            return []
        
        replies = data.get("data", {}).get("replies", [])
        comments = []
        
        for reply in replies:
            comment = {
                "user": reply.get("member", {}).get("uname", "未知用户"),
                "content": reply.get("content", {}).get("message", "").replace("\n", " "),
                "like": reply.get("like", 0),
                "time": reply.get("ctime", 0)
            }
            comments.append(comment)
        
        return comments
        
    except Exception as e:
        print(f"获取评论异常 (aid={aid}): {e}") 
        return []

# ==========================================
# 第二部分：数据分析模块
# ==========================================

def analyze_partition_data(videos):
    """
    分析各分区视频的数量占比与平均互动率
    :param videos: 排行榜视频列表
    :return: DataFrame
    """
    partition_stats = {}
    for v in videos:
        tname = v.get('tname', '未知分区')
        stat = v.get('stat', {})
        
        if tname not in partition_stats:
            partition_stats[tname] = {
                'count': 0,
                'total_view': 0,
                'total_like': 0,
                'total_coin': 0,
                'total_reply': 0
            }
        
        partition_stats[tname]['count'] += 1
        partition_stats[tname]['total_view'] += stat.get('view', 0)
        partition_stats[tname]['total_like'] += stat.get('like', 0)
        partition_stats[tname]['total_coin'] += stat.get('coin', 0)
        partition_stats[tname]['total_reply'] += stat.get('reply', 0)
    
    df_data = []
    for tname, stats in partition_stats.items():
        view = stats['total_view']
        like_rate = stats['total_like'] / view if view > 0 else 0
        coin_rate = stats['total_coin'] / view if view > 0 else 0
        reply_rate = stats['total_reply'] / view if view > 0 else 0
        interaction_rate = (stats['total_like'] + stats['total_coin'] + stats['total_reply']) / view if view > 0 else 0
        
        df_data.append({
            '分区': tname,
            '视频数量': stats['count'],
            '占比(%)': stats['count'] / len(videos) * 100,
            '点赞率(%)': like_rate * 100,
            '投币率(%)': coin_rate * 100,
            '评论率(%)': reply_rate * 100,
            '综合互动率(%)': interaction_rate * 100
        })
    
    df = pd.DataFrame(df_data).sort_values('视频数量', ascending=False)
    return df

class CommentAnalyzer:
    """评论多维度分析器"""
    
    def __init__(self):
        self.praise_words = ['好', '赞', '喜欢', '支持', '不错', '优秀', '厉害', '牛逼', '爱了', '推荐']
        self.criticize_words = ['差', '烂', '垃圾', '失望', '无语', '坑', '抄袭', '水', '恶心']
        self.question_words = ['怎么', '如何', '为什么', '？', '吗', '呢', '啥', '谁', '哪']
    
    def analyze_sentiment(self, text):
        """简单情感倾向分析（基于关键词）"""
        if not text or len(text) < 2:
            return {'polarity': 0, 'sentiment': 'neutral'}
        
        text = text.lower()
        praise_score = sum(1 for w in self.praise_words if w in text)
        criticize_score = sum(1 for w in self.criticize_words if w in text)
        
        polarity = 0
        if praise_score > criticize_score:
            polarity = min(0.5 + (praise_score - criticize_score) * 0.1, 0.9)
            sentiment = 'positive'
        elif criticize_score > praise_score:
            polarity = -min(0.5 + (criticize_score - praise_score) * 0.1, 0.9)
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {'polarity': round(polarity, 3), 'sentiment': sentiment}
    
    def classify_intent(self, text):
        """评论意图分类"""
        if not text:
            return 'other'
        
        if any(word in text for word in self.praise_words):
            return 'praise'
        if any(word in text for word in self.criticize_words):
            return 'criticize'
        if any(word in text for word in self.question_words):
            return 'question'
        if '建议' in text or '希望' in text or '应该' in text:
            return 'suggestion'
        if '分享' in text or '推荐' in text or '安利' in text:
            return 'sharing'
        return 'other'
    
    def assess_quality(self, text):
        """评论质量评估"""
        if not text:
            return {'score': 0, 'quality_level': '低'}
        
        score = 0
        length = len(text)
        if 20 <= length <= 100:
            score += 0.4
        elif length > 100:
            score += 0.3
        else:
            score += 0.1
        
        if re.search(r'\[.*?\]', text) or re.search(r'[\U0001F600-\U0001F64F]', text):
            score += 0.2
        if re.search(r'\d+', text) or re.search(r'[A-Za-z]{2,}', text):
            score += 0.2
        if len(set(text)) > 10:
            score += 0.2
        
        score = max(0, min(1, score))
        if score >= 0.6:
            quality_level = '高'
        elif score >= 0.3:
            quality_level = '中'
        else:
            quality_level = '低'
        
        return {'score': round(score, 2), 'quality_level': quality_level}
    
    def analyze_comments(self, comments):
        """批量分析评论"""
        results = []
        for comment in comments:
            text = comment.get('content', '')
            if not text:
                continue
            
            sentiment = self.analyze_sentiment(text)
            intent = self.classify_intent(text)
            quality = self.assess_quality(text)
            
            results.append({
                **comment,
                'sentiment_polarity': sentiment['polarity'],
                'sentiment_category': sentiment['sentiment'],
                'intent': intent,
                'quality_score': quality['score'],
                'quality_level': quality['quality_level']
            })
        return results

def visualize_extended_data(top_creators, analyzed_comments, videos):
    """
    可视化扩展数据挖掘结果
    :param top_creators: TOP创作者列表
    :param analyzed_comments: 分析后的评论列表
    :param videos: 视频列表
    """
    if not top_creators and not analyzed_comments and not videos:
        print("无扩展数据可可视化")
        return
    
    # 确保中文字体配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('扩展数据挖掘可视化', fontsize=18, fontweight='bold')
    
    # 1. TOP创作者影响力 (左上)
    ax1 = plt.subplot(2, 2, 1)
    if top_creators:
        names = [c['name'] for c in top_creators]
        views = [c['total_view'] / 10000 for c in top_creators]  # 转换为万
        colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, len(names)))[::-1]
        bars = ax1.barh(names, views, color=colors_bar)
        ax1.set_xlabel('总播放量 (万)', fontsize=11)
        ax1.set_title('TOP5创作者影响力', fontsize=13, fontweight='bold')
        # 添加数值标签
        for i, (bar, view) in enumerate(zip(bars, views)):
            ax1.text(view + 5, bar.get_y() + bar.get_height()/2, 
                    f'{view:.1f}万', va='center', fontsize=10)
    
    # 2. 评论时段分布 (右上)
    ax2 = plt.subplot(2, 2, 2)
    if analyzed_comments:
        df_comm = pd.DataFrame(analyzed_comments)
        df_comm['date'] = pd.to_datetime(df_comm['time'], unit='s')
        df_comm['hour'] = df_comm['date'].dt.hour
        hour_counts = df_comm['hour'].value_counts().sort_index()
        
        # 填补缺失的小时
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(hour_counts)
        
        ax2.plot(all_hours.index, all_hours.values, marker='o', linewidth=2, 
                markersize=8, color='#FF6B6B')
        ax2.fill_between(all_hours.index, all_hours.values, alpha=0.3, color='#FF6B6B')
        ax2.set_xlabel('小时', fontsize=11)
        ax2.set_ylabel('评论数量', fontsize=11)
        ax2.set_title('评论时段分布', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        
        # 标记峰值
        max_hour = all_hours.idxmax()
        max_count = all_hours.max()
        ax2.annotate(f'峰值: {max_hour}:00', 
                    xy=(max_hour, max_count),
                    xytext=(max_hour + 1, max_count + 2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    # 3. 视频发布星期分布 (左下)
    ax3 = plt.subplot(2, 2, 3)
    if videos:
        pubdates = [v.get('pubdate', 0) for v in videos if v.get('pubdate')]
        if pubdates:
            dates = [datetime.fromtimestamp(t) for t in pubdates]
            weekdays = [d.strftime('%A') for d in dates]
            weekday_counts = Counter(weekdays)
            
            # 按星期排序
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_cn = {
                'Monday': '周一', 'Tuesday': '周二', 'Wednesday': '周三',
                'Thursday': '周四', 'Friday': '周五', 'Saturday': '周六', 'Sunday': '周日'
            }
            
            ordered_counts = []
            ordered_labels = []
            for day in weekday_order:
                if day in weekday_counts:
                    ordered_counts.append(weekday_counts[day])
                    ordered_labels.append(weekday_cn[day])
            
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(ordered_counts)))
            wedges, texts, autotexts = ax3.pie(ordered_counts, labels=ordered_labels, 
                                                autopct='%1.1f%%', colors=colors_pie,
                                                textprops={'fontsize': 11})
            ax3.set_title('视频发布星期分布', fontsize=13, fontweight='bold')
            
            # 突出显示最高的扇区
            max_idx = ordered_counts.index(max(ordered_counts))
            wedges[max_idx].set_edgecolor('red')
            wedges[max_idx].set_linewidth(2)
            wedges[max_idx].set_hatch('/')
    
    # 4. 综合数据卡片 (右下)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # 收集统计数据
    stats_text = "扩展数据统计\n\n"
    
    if top_creators:
        stats_text += f"TOP创作者数: {len(top_creators)}\n"
        stats_text += f"最高播放: {top_creators[0]['total_view']:,}\n"
        stats_text += f"平均视频数/人: {sum(c['video_count'] for c in top_creators)/len(top_creators):.1f}\n\n"
    
    if analyzed_comments:
        df_comm = pd.DataFrame(analyzed_comments)
        df_comm['date'] = pd.to_datetime(df_comm['time'], unit='s')
        
        stats_text += f"总评论数: {len(analyzed_comments)}\n"
        stats_text += f"评论跨度: {df_comm['date'].min().strftime('%Y-%m-%d')} ~ {df_comm['date'].max().strftime('%Y-%m-%d')}\n"
        
        # 计算评论活跃时段
        df_comm['hour'] = df_comm['date'].dt.hour
        peak_hour = df_comm['hour'].mode().iloc[0] if not df_comm['hour'].mode().empty else 0
        stats_text += f"最活跃时段: {peak_hour:02d}:00\n\n"
    
    if videos:
        stats_text += f"视频总数: {len(videos)}\n"
        
        # 分区统计
        partitions = [v.get('tname', '未知') for v in videos]
        top_partition = Counter(partitions).most_common(1)[0]
        stats_text += f"最多分区: {top_partition[0]} ({top_partition[1]}个)\n"
        
        # 平均数据
        avg_view = np.mean([v.get('stat', {}).get('view', 0) for v in videos])
        stats_text += f"平均播放: {avg_view:,.0f}"
    
    # 显示统计文本
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()

def extended_data_mining(videos, analyzed_comments):
    """扩展数据挖掘（增强版）"""
    print("\n" + "=" * 80)
    print("扩展数据挖掘")
    print("=" * 80)
    
    # 1. 创作者影响力分析
    creator_stats = {}
    for v in videos:
        owner = v.get('owner', {})
        mid = owner.get('mid')
        name = owner.get('name', '未知')
        stat = v.get('stat', {})
        
        if mid not in creator_stats:
            creator_stats[mid] = {'name': name, 'video_count': 0, 'total_view': 0}
        creator_stats[mid]['video_count'] += 1
        creator_stats[mid]['total_view'] += stat.get('view', 0)
    
    top_creators = sorted(creator_stats.values(), key=lambda x: x['total_view'], reverse=True)[:5]
    print("\n榜单TOP创作者 (按总播放量):")
    for i, c in enumerate(top_creators, 1):
        print(f"  {i}. {c['name']}: {c['video_count']}个视频, 总播放 {c['total_view']:,}")
    
    # 2. 评论时段分布（修正输出逻辑）
    print("\n评论活跃时段 (Top 5):")
    if analyzed_comments:
        df_comm = pd.DataFrame(analyzed_comments)
        df_comm['date'] = pd.to_datetime(df_comm['time'], unit='s')
        df_comm['hour'] = df_comm['date'].dt.hour
        
        # 计算每个小时的评论数量
        hour_counts = df_comm['hour'].value_counts().sort_index()
        
        # 显示评论数量最多的前5个时段（按评论数量排序）
        print("  按评论数量排序的Top 5活跃时段:")
        top_5_hours = hour_counts.sort_values(ascending=False).head(5)
        for hour, count in top_5_hours.items():
            print(f"  {hour:02d}:00-{hour+1:02d}:00  {count}条评论")
        
        # 也显示完整的时段分布（可选）
        print("\n  完整时段分布:")
        for hour, count in hour_counts.items():
            if count > 0:  # 只显示有评论的时段
                bar = '█' * (count // 2)  # 简单的条形图
                print(f"  {hour:02d}:00-{hour+1:02d}:00  {count:3d}条 {bar}")
    
    # 3. 视频发布星期分布
    print("\n视频发布星期分布:")
    pubdates = [v.get('pubdate', 0) for v in videos if v.get('pubdate')]
    if pubdates:
        dates = [datetime.fromtimestamp(t) for t in pubdates]
        weekdays = [d.strftime('%A') for d in dates]
        weekday_cn = {
            'Monday': '周一', 'Tuesday': '周二', 'Wednesday': '周三',
            'Thursday': '周四', 'Friday': '周五', 'Saturday': '周六', 'Sunday': '周日'
        }
        weekday_counts = Counter(weekdays)
        # 按星期排序显示
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in weekday_order:
            if day in weekday_counts:
                cn_day = weekday_cn.get(day, day)
                print(f"  {cn_day}: {weekday_counts[day]}个视频")
    
    # 4. 可视化扩展数据
    print("\n生成扩展数据可视化图表...")
    visualize_extended_data(top_creators, analyzed_comments, videos)
    
    return top_creators

# ==========================================
# 第三部分：可视化模块
# ==========================================

def visualize_partition_data(df):
    """可视化分区数据"""
    if df.empty:
        print("无分区数据可可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('B站排行榜分区数据分析', fontsize=16, fontweight='bold')
    
    # 1. 分区数量占比（饼图）
    ax1 = axes[0, 0]
    top8 = df.head(8)
    other_count = df['视频数量'].sum() - top8['视频数量'].sum()
    if other_count > 0:
        labels = top8['分区'].tolist() + ['其他']
        sizes = top8['视频数量'].tolist() + [other_count]
    else:
        labels = top8['分区'].tolist()
        sizes = top8['视频数量'].tolist()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax1.set_title('各分区视频数量占比')
    
    # 2. 分区视频数量排行
    ax2 = axes[0, 1]
    top10 = df.head(10)
    ax2.barh(top10['分区'], top10['视频数量'], color='skyblue')
    ax2.set_xlabel('视频数量')
    ax2.set_title('各分区视频数量排行 (Top 10)')
    
    # 3. 综合互动率排行
    ax3 = axes[1, 0]
    top_interaction = df.sort_values('综合互动率(%)', ascending=False).head(10)
    ax3.barh(top_interaction['分区'], top_interaction['综合互动率(%)'], color='lightcoral')
    ax3.set_xlabel('综合互动率 (%)')
    ax3.set_title('各分区综合互动率排行 (Top 10)')
    
    # 4. 视频数量 vs 互动率
    ax4 = axes[1, 1]
    ax4.scatter(df['视频数量'], df['综合互动率(%)'], s=80, alpha=0.7, color='green')
    for _, row in df.iterrows():
        ax4.annotate(row['分区'], (row['视频数量'], row['综合互动率(%)']), fontsize=8, alpha=0.7)
    ax4.set_xlabel('视频数量')
    ax4.set_ylabel('综合互动率 (%)')
    ax4.set_title('视频数量 vs 互动率')
    
    plt.tight_layout()
    plt.show()

def visualize_comment_analysis(df_comments):
    """可视化评论分析结果"""
    if df_comments is None or df_comments.empty:
        print("无评论数据可可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('评论多维度分析', fontsize=16, fontweight='bold')
    
    # 1. 情感倾向分布
    ax1 = axes[0, 0]
    sentiment_counts = df_comments['sentiment_category'].value_counts()
    colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=[colors.get(k, '#999') for k in sentiment_counts.index])
    ax1.set_title('情感倾向分布')
    
    # 2. 意图分布
    ax2 = axes[0, 1]
    intent_map = {'praise': '赞扬', 'criticize': '批评', 'question': '提问',
                  'suggestion': '建议', 'sharing': '分享', 'other': '其他'}
    intent_counts = df_comments['intent'].value_counts()
    labels = [intent_map.get(i, i) for i in intent_counts.index]
    ax2.bar(labels, intent_counts.values, color='#42A5F5')
    ax2.set_xlabel('意图类型')
    ax2.set_ylabel('数量')
    ax2.set_title('评论意图分布')
    
    # 3. 质量等级分布
    ax3 = axes[1, 0]
    quality_counts = df_comments['quality_level'].value_counts()
    colors_q = {'高': '#66BB6A', '中': '#FFA726', '低': '#EF5350'}
    ax3.bar(quality_counts.index, quality_counts.values, 
            color=[colors_q.get(k, '#999') for k in quality_counts.index])
    ax3.set_xlabel('质量等级')
    ax3.set_ylabel('数量')
    ax3.set_title('评论质量等级分布')
    
    # 4. 质量评分直方图
    ax4 = axes[1, 1]
    ax4.hist(df_comments['quality_score'], bins=10, color='#AB47BC', edgecolor='white')
    ax4.axvline(df_comments['quality_score'].mean(), color='red', linestyle='--', label=f'均值: {df_comments["quality_score"].mean():.2f}')
    ax4.set_xlabel('质量评分')
    ax4.set_ylabel('频率')
    ax4.set_title('评论质量评分分布')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 第四部分：主程序
# ==========================================

def main():
    """主程序"""
    print("=" * 80)
    print("Bilibili 排行榜数据分析系统")
    print("=" * 80)
    
    # Step 1: 爬取排行榜数据
    print("\nStep 1: 爬取排行榜数据...")
    videos = get_rank_list()
    if not videos:
        print("数据获取失败，程序退出")
        return
    
    # Step 2: 爬取评论数据（每个视频取前10条热门评论）
    print("\nStep 2: 爬取视频评论...")
    all_comments = {}
    for i, v in enumerate(videos[:50], 1):
        aid = v.get('aid')
        if aid:
            print(f"  正在获取第 {i} 个视频的评论...")
            comments = get_hot_comments(aid, limit=10)
            all_comments[aid] = comments
            time.sleep(1)  # 礼貌延迟
    
    # Step 3: 分析分区数据
    print("\nStep 3: 分析分区数据...")
    partition_df = analyze_partition_data(videos)
    print("\n分区统计结果:")
    df_display = partition_df.round(2)
    for col in df_display.select_dtypes(include=['float64', 'int64']).columns:
        df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")
    print_table(df_display)
    
    # Step 4: 分析评论数据
    print("\nStep 4: 分析评论数据...")
    analyzer = CommentAnalyzer()
    all_analyzed = []
    for aid, comments in all_comments.items():
        if comments:
            analyzed = analyzer.analyze_comments(comments)
            all_analyzed.extend(analyzed)
    
    if all_analyzed:
        df_comments = pd.DataFrame(all_analyzed)
        print(f"共分析 {len(df_comments)} 条评论")
        print("\n情感分布:")
        print(df_comments['sentiment_category'].value_counts())
        print("\n意图分布:")
        print(df_comments['intent'].value_counts())
        print("\n质量等级分布:")
        print(df_comments['quality_level'].value_counts())
    else:
        df_comments = None
        print("没有获取到有效评论")
    
    # Step 5: 扩展数据挖掘（包含可视化）
    print("\nStep 5: 扩展数据挖掘...")
    top_creators = extended_data_mining(videos, all_analyzed if all_analyzed else [])
    
    # Step 6: 可视化（只对分区和评论数据，扩展数据已在Step 5中可视化）
    print("\nStep 6: 生成基础可视化图表...")
    visualize_partition_data(partition_df)
    if df_comments is not None and not df_comments.empty:
        visualize_comment_analysis(df_comments)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
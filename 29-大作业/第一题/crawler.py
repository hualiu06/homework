# -*- coding: utf-8 -*-
"""
=================================================================
大连理工大学《Python与智能数据分析》课程大作业 —— 作业一
爬虫脚本：从红黑统计公报库爬取辽宁省14个地级市人口数据

运行方法：
    python crawler.py

运行前请先安装依赖：
    pip install requests beautifulsoup4 pandas lxml

注意：本脚本需要联网运行！
=================================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 输出目录
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# ============================================================
# 第一步：自动获取辽宁省14个地级市ID
# ============================================================
print("=" * 60)
print("爬虫启动 —— 数据来源：红黑统计公报库 (hongheiku.com)")
print("=" * 60)
print("\n【步骤1】自动获取辽宁省14个地级市ID...")

def get_liaoning_city_ids():
    """从辽宁标签页面自动获取所有地级市的URL ID"""
    url = "https://www.hongheiku.com/tag/%E8%BE%BD%E5%AE%81"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    city_links = {}
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if '/shijirenkou/' in href and text.endswith('市'):
            city_name = text.replace('市', '').replace('[辽宁]', '').strip()
            match = re.search(r'/shijirenkou/(\d+)\.html', href)
            if match:
                city_id = match.group(1)
                if city_name not in city_links:
                    city_links[city_name] = city_id
    return city_links

city_ids = get_liaoning_city_ids()
print(f"成功获取 {len(city_ids)} 个城市:")
for city in sorted(city_ids.keys()):
    print(f"  {city} (ID: {city_ids[city]})")

# ============================================================
# 第二步：爬取每个城市主页面的人口数据
# ============================================================
print("\n【步骤2】爬取各城市人口数据...")

def extract_population_from_main(city_name, city_id):
    """
    从城市主页面提取人口数据。
    方法：requests请求 → BeautifulSoup解析 → 正则提取结构化数据
    
    页面结构：
    - 每年数据以 <strong>XXXX年末XXXX年初</strong>,数据... 格式存储
    - 部分城市使用 <strong>XXXX末XXXX初</strong> 格式
    """
    url = f"https://www.hongheiku.com/shijirenkou/{city_id}.html"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = 'utf-8'
        html = resp.text
        
        records = []
        
        # 正则1: <strong>2025年末2026年初</strong>,数据...
        pattern1 = r'<strong>(\d{4})年末(\d{4})年初</strong>,?([^<]+(?:<(?!strong>)[^<]*)*?)(?=<strong>|<h2|<h3|<div class="entry-header")'
        matches1 = re.findall(pattern1, html, re.DOTALL)
        
        # 正则2: <strong>2019末2020初</strong>，数据...
        pattern2 = r'<strong>(\d{4})末(\d{4})初</strong>[，,]?([^<]+(?:<(?!strong>)[^<]*)*?)(?=<strong>|<h2|<h3|<div class="entry-header")'
        matches2 = re.findall(pattern2, html, re.DOTALL)
        
        # 合并去重
        all_matches = matches1 + matches2
        seen_years = set()
        
        for y1, y2, content in all_matches:
            year1, year2 = int(y1), int(y2)
            data_year = max(year1, year2) - 1
            
            if data_year in seen_years:
                continue
            seen_years.add(data_year)
            
            clean = re.sub(r'<[^>]+>', '', content).strip()
            clean = re.sub(r'\s+', ' ', clean)
            
            record = {'城市': city_name, '年份': data_year}
            
            # 提取各项指标
            for key, regex in [
                ('常住人口_万人', r'常住人口([\d.]+)\s*万'),
                ('城镇化率_%', r'城镇化率([\d.]+)%'),
                ('出生率_‰', r'出生率([\d.\-]+)\s*‰'),
                ('死亡率_‰', r'死亡率([\d.\-]+)\s*‰'),
                ('自然增长率_‰', r'自然增长率([\d.\-]+)\s*‰'),
                ('户籍人口_万人', r'户籍人口([\d.]+)\s*万'),
                ('户籍60岁以上_万人', r'60岁以上([\d.]+)\s*万'),
            ]:
                m = re.search(regex, clean)
                if m:
                    record[key] = float(m.group(1))
            
            if '常住人口_万人' in record:
                records.append(record)
        
        print(f"  {city_name}: 主页面获取 {len(records)} 条")
        return records
    
    except Exception as e:
        print(f"  {city_name}: 主页面失败 ({e})")
        return []

# ============================================================
# 第三步：从"历史数据"子页面补充爬取
# ============================================================
print("\n【步骤3】从历史数据页面补充爬取...")

def extract_from_history_pages(city_name, city_id):
    """
    从城市的历史数据页面(lishishuju)补充爬取。
    每个城市的历史数据页面链接在主页面的"历史相关数据"部分。
    
    URL格式: https://www.hongheiku.com/lishishuju/{id}.html
    这些页面是静态HTML，包含更详细的年份信息。
    """
    # 先从主页面找到所有 lishishuju 链接
    main_url = f"https://www.hongheiku.com/shijirenkou/{city_id}.html"
    
    try:
        resp = requests.get(main_url, headers=HEADERS, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 找所有历史数据链接
        history_urls = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if 'lishishuju' in href and text.startswith('20'):
                history_urls.append(href)
        
        records = []
        seen_years = set()
        
        for h_url in history_urls[:10]:  # 最多爬10个历史页面
            try:
                resp2 = requests.get(h_url, headers=HEADERS, timeout=10)
                resp2.encoding = 'utf-8'
                soup2 = BeautifulSoup(resp2.text, 'html.parser')
                text2 = soup2.get_text(separator='\n')
                
                # 从文本中提取数据
                for line in text2.split('\n'):
                    line = line.strip()
                    if len(line) < 20:
                        continue
                    
                    # 提取年份
                    year_match = re.search(r'(\d{4})年末?(\d{4})年初?', line)
                    if not year_match:
                        year_match = re.search(r'(\d{4})末(\d{4})初', line)
                    if not year_match:
                        continue
                    
                    y1, y2 = int(year_match.group(1)), int(year_match.group(2))
                    data_year = max(y1, y2) - 1
                    
                    if data_year in seen_years or data_year < 2014:
                        continue
                    
                    record = {'城市': city_name, '年份': data_year}
                    
                    for key, regex in [
                        ('常住人口_万人', r'常住人口([\d.]+)\s*万'),
                        ('城镇化率_%', r'城镇化率([\d.]+)%'),
                        ('出生率_‰', r'出生率([\d.\-]+)\s*‰'),
                        ('死亡率_‰', r'死亡率([\d.\-]+)\s*‰'),
                        ('自然增长率_‰', r'自然增长率([\d.\-]+)\s*‰'),
                        ('户籍人口_万人', r'户籍人口([\d.]+)\s*万'),
                        ('户籍60岁以上_万人', r'60岁以上([\d.]+)\s*万'),
                    ]:
                        m = re.search(regex, line)
                        if m:
                            record[key] = float(m.group(1))
                    
                    if '常住人口_万人' in record:
                        records.append(record)
                        seen_years.add(data_year)
                
                time.sleep(0.5)
            except:
                continue
        
        if records:
            print(f"  {city_name}: 历史页面补充 {len(records)} 条")
        return records
    
    except:
        return []

# 执行爬取
all_records = []
for city_name in sorted(city_ids.keys()):
    city_id = city_ids[city_name]
    
    # 从主页面爬取
    main_records = extract_population_from_main(city_name, city_id)
    all_records.extend(main_records)
    time.sleep(1)
    
    # 从历史页面补充
    hist_records = extract_from_history_pages(city_name, city_id)
    all_records.extend(hist_records)
    time.sleep(0.5)

df = pd.DataFrame(all_records)
# 去重（同一城市同一年份保留最新的）
if not df.empty:
    df = df.sort_values('年份', ascending=False).drop_duplicates(subset=['城市', '年份'], keep='first')
    df = df.sort_values(['城市', '年份']).reset_index(drop=True)

print(f"\n人口数据汇总: 共 {len(df)} 条, {df['城市'].nunique()} 个城市")
if not df.empty:
    print(f"年份范围: {df['年份'].min()}-{df['年份'].max()}")

# ============================================================
# 第四步：从主页面信息摘要中提取最新GDP和收入
# ============================================================
print("\n【步骤4】从城市主页提取GDP和收入信息...")

gdp_income_records = []
for city_name in sorted(city_ids.keys()):
    city_id = city_ids[city_name]
    url = f"https://www.hongheiku.com/shijirenkou/{city_id}.html"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        text = resp.text
        
        # 从页面摘要中提取最新GDP和收入
        # 格式: "9100.3亿2025年GDP数据"
        record = {'城市': city_name}
        
        gdp_match = re.search(r'([\d.]+)亿\d{4}年GDP数据', text)
        if gdp_match:
            record['最新GDP_亿元'] = float(gdp_match.group(1))
        
        income_match = re.search(r'(\d+)元\d{4}年城镇居民可支配收入', text)
        if income_match:
            record['城镇居民可支配收入_元'] = float(income_match.group(1))
        
        if '最新GDP_亿元' in record or '城镇居民可支配收入_元' in record:
            gdp_income_records.append(record)
        
        time.sleep(0.5)
    except:
        continue

df_gdp_income = pd.DataFrame(gdp_income_records)
if not df_gdp_income.empty:
    print(f"GDP/收入数据: {len(df_gdp_income)} 个城市")

# ============================================================
# 第五步：数据合并与计算
# ============================================================
print("\n【步骤5】数据合并与计算...")

if not df.empty:
    # 计算老龄化率
    mask = (df['户籍60岁以上_万人'].notna()) & (df['户籍人口_万人'].notna()) & (df['户籍人口_万人'] > 0)
    df.loc[mask, '老龄化率_%'] = (df.loc[mask, '户籍60岁以上_万人'] / df.loc[mask, '户籍人口_万人']) * 100
    
    # 选择最终列
    keep_cols = ['城市', '年份', '常住人口_万人', '出生率_‰', '死亡率_‰', '自然增长率_‰',
                 '城镇化率_%', '老龄化率_%']
    existing_cols = [c for c in keep_cols if c in df.columns]
    df_final = df[existing_cols].copy()
    
    # 保存
    csv_path = os.path.join(DATA_DIR, '辽宁省人口数据_爬取.csv')
    df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 显示结果
    print(f"\n{'='*60}")
    print(f"爬取完成！数据预览:")
    print(f"{'='*60}")
    print(df_final.to_string())
    
    print(f"\n各城市数据覆盖情况:")
    for city in sorted(df_final['城市'].unique()):
        cd = df_final[df_final['城市'] == city]
        years = sorted(cd['年份'].tolist())
        has_complete = cd[['常住人口_万人', '出生率_‰', '死亡率_‰']].notna().all(axis=1).sum()
        print(f"  {city}: {len(cd)}年 ({years[0] if years else '?'}-{years[-1] if years else '?'}) 完整指标: {has_complete}条")
    
    print(f"\n数据已保存至: {csv_path}")
    
    # 保存GDP收入数据
    if not df_gdp_income.empty:
        gdp_path = os.path.join(DATA_DIR, '辽宁省GDP收入数据_爬取.csv')
        df_gdp_income.to_csv(gdp_path, index=False, encoding='utf-8-sig')
        print(f"GDP/收入数据保存至: {gdp_path}")
    
    print(f"\n✅ 爬虫完成！请继续运行 analysis.py 进行数据分析")
else:
    print("\n❌ 爬取失败！请检查网络连接后重试。")

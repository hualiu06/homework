"""
猎聘网 AI 岗位爬虫
===================
采集中国主要城市 AI 相关岗位招聘信息

输出字段: 岗位名称, 城市, 月薪, 学历要求, 经验要求, 公司, 行业, 公司规模
输出文件: liepin_ai_jobs.csv

使用方法:
  pip install selenium beautifulsoup4 lxml
  python liepin_crawler.py

注意: 浏览器窗口会自动打开，如遇验证码请手动通过。
"""

import os, sys, time, re, random, csv, json
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
except ImportError:
    print("请先安装 selenium: pip install selenium")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("请先安装 beautifulsoup4: pip install beautifulsoup4")
    sys.exit(1)

# 输出文件路径（与脚本同目录）
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liepin_ai_jobs.csv')
# Edge 浏览器路径
EDGE_PATH = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

# AI 相关关键词
# 爬虫会遍历这些关键词，逐个搜索猎聘网
KEYWORDS = [
    'AI工程师', '人工智能', '机器学习', '深度学习', '大模型',
    '算法工程师', 'NLP', '计算机视觉', '推荐算法', '数据科学家',
    'AIGC', 'LLM', '大语言模型', '自然语言处理', 'AI产品经理',
]

MAX_PAGES_PER_KW = 5    # 每个关键词最多翻 5 页
PAGE_SIZE = 40          # 每页显示 40 条结果
DELAY = (4, 8)          # 两次请求之间的随机延迟范围（4~8 秒，防反爬）

# ============ 工具函数 ============

def log(msg):
    """打印带时间戳的日志"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    except UnicodeEncodeError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.encode('ascii', errors='replace').decode('ascii')}")


def randsleep():
    """随机等待一段时间，模拟人类操作，降低被反爬检测的风险"""
    time.sleep(random.uniform(*DELAY))


def create_driver():
    """创建 Edge 浏览器实例（非无头模式，用户可见以便手动处理验证码）"""
    options = EdgeOptions()
    options.binary_location = EDGE_PATH
    # 隐藏自动化控制特征，防止被网站检测为爬虫
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--start-maximized')        # 窗口最大化
    options.add_argument('--disable-notifications')   # 禁用通知弹窗

    driver = webdriver.Edge(options=options)

    # 注入 JavaScript 代码，覆盖 navigator 的属性，进一步伪装成真实浏览器
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': """
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
    """})
    return driver


def parse_salary(text):
    """
    从文本中提取月薪范围（元）
    示例: '50-80k' -> (50000, 80000), '薪资面议' -> (None, None)

    处理逻辑:
    1. 先排除 "面议"/"面谈"
    2. 按 "·" 或 "×" 切分，只取前半部分（忽略 "14薪" 这类后缀）
    3. 匹配数字范围，如果有 "k" 单位则乘以 1000
    """
    if not text or '面议' in text or '面谈' in text:
        return None, None
    # 去掉 "·14薪" 这类后缀信息
    text = re.split(r'[·×]', text)[0].strip()
    # 匹配 "35-55k" 或 "35~55k" 格式
    m = re.search(r'(\d+(?:\.\d+)?)\s*k?\s*[-–至~]\s*(\d+(?:\.\d+)?)\s*k?', text, re.I)
    if m:
        v1, v2 = float(m.group(1)), float(m.group(2))
        if 'k' in text.lower() or (v1 < 100 and v2 < 100):
            return min(v1, v2) * 1000, max(v1, v2) * 1000
        return min(v1, v2), max(v1, v2)
    # 如果没匹配到范围，尝试提取单个数字
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if m:
        v = float(m.group(1))
        if 'k' in text.lower() or v < 100:
            return v * 1000, v * 1000
        return v, v
    return None, None


def parse_experience(text):
    """
    标准化工作经验要求
    返回 (最小值, 最大值)，如 "5-10年" -> (5, 10)
    "经验不限" -> (0, 20)，"应届" -> (0, 0)
    """
    if not text:
        return None, None
    text = text.strip()
    if '不限' in text:
        return 0, 20
    if '应届' in text:
        return 0, 0
    # 匹配 "5-10年" 格式
    m = re.search(r'(\d+)\s*[-–至~]\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    # 匹配 "3年以上" 格式，上限设为 99
    m = re.search(r'(\d+)\s*年?\s*以上', text)
    if m:
        return int(m.group(1)), 99
    return None, None


def parse_education(text):
    """标准化学历要求，将各种表述映射到统一的高、中、低三类"""
    if not text:
        return '不限'
    if '博' in text: return '博士'
    if '硕' in text: return '硕士'
    if '本' in text: return '本科'
    if '大专' in text: return '大专'
    if '不限' in text: return '不限'
    return text


def extract_skills_from_title(title):
    """从岗位名称中提取技能关键词（如标题包含 "NLP" 就标记为 NLP 技能）"""
    skill_map = {
        'python': 'Python', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow',
        'nlp': 'NLP', 'llm': 'LLM/大模型', 'aigc': 'AIGC',
        'cv': '计算机视觉', 'vision': '计算机视觉',
        'recommend': '推荐系统', 'rec': '推荐系统',
        'search': '搜索', 'speech': '语音', 'audio': '语音',
        'rag': 'RAG', 'langchain': 'LangChain',
        'transformer': 'Transformer', 'diffusion': '扩散模型',
        'gpt': 'GPT', 'bert': 'BERT', 'deep learning': '深度学习',
        'machine learning': '机器学习', 'data mining': '数据挖掘',
    }
    found = []
    tl = title.lower()
    for key, val in skill_map.items():
        if key in tl:
            found.append(val)
    return list(set(found))


def handle_verification(driver):
    """等待页面加载完成（预留接口，后续可扩展验证码处理逻辑）"""
    time.sleep(4)


def extract_job_cards(soup):
    """从页面 HTML 中提取所有岗位卡片容器"""
    # 优先使用猎聘网的标准卡片选择器
    cards = soup.select('.job-card-pc-container')
    if not cards:
        # 备用方案：如果标准选择器没匹配到，尝试模糊匹配其他可能的卡片类名
        cards = soup.select('[class*="job-card"], [class*="job-item"], li[class*="job"]')
    return cards


# AI 关键词列表，用于过滤非相关岗位，避免采集到不相关的职位
AI_KEYWORDS = [
    'ai', '人工智能', '算法', '机器学习', '深度学习', '大模型', 'llm',
    'nlp', '自然语言', '计算机视觉', 'cv', '推荐', '搜索',
    'aigc', '数据科学', '数据挖掘', '数据分析', '数据开发',
    'pytorch', 'tensorflow', 'gpt', 'bert', 'transformer',
    'langchain', 'rag', 'agent', '多模态', '语音',
    '自动驾驶', '机器人', '图像', '识别',
    'ai产品', 'ai运营', 'ai应用', '智能',
]


def is_ai_related(title):
    """判断岗位名称是否与 AI 相关（标题中包含 AI_KEYWORDS 中的任意一个）"""
    tl = title.lower()
    for kw in AI_KEYWORDS:
        if kw in tl:
            return True
    return False


def parse_card(card, keyword, city_name):
    """解析单个岗位卡片，提取结构化信息

    猎聘网卡片的标准 DOM 结构（已验证）:
      [0]  岗位名称
      [1]  【
      [2]  城市（如 '北京-海淀区'）
      [3]  】
      [4]  薪资（如 '35-55k·14薪'）
      [5]  经验要求（如 '5-10年'）
      [6]  学历要求（如 '本科'、'硕士'）
      [7]  公司名称
      [8]  行业
      [9]  （可选）融资阶段（如 'A轮'、'已上市'）
      [10] 公司规模（或 [9] 如果没有融资信息）
      [11] 招聘者信息
      [12] （可选）活跃时间

    解析策略：通过位置和内容特征综合判断各字段的位置，而非死板地按固定索引，
    因为部分卡片可能缺少某些字段（如融资阶段）。
    """
    texts = list(card.stripped_strings)
    if not texts:
        return None
    n = len(texts)

    # 初始化岗位信息字典，所有字段都有默认值
    job = {
        'job_title': texts[0] if n > 0 else '',
        'city': '',
        'salary_text': '',
        'salary_min': None,
        'salary_max': None,
        'education': '不限',
        'experience_text': '',
        'exp_min': None,
        'exp_max': None,
        'company': '',
        'industry': '',
        'company_size': '',
        'funding_stage': '',
        'employment_type': '',
        'skills': '',
        'keyword': keyword,          # 搜索该岗位时使用的关键词
        'city_raw': city_name,       # 原始城市信息
        'source': 'liepin',          # 数据来源标记
        'scrape_date': datetime.now().strftime('%Y-%m-%d'),  # 爬取日期
    }

    # ----- 逐字段解析 -----

    # 岗位名称（第 0 个元素）
    if n > 0:
        job['job_title'] = texts[0]

    # 城市（第 2 个元素，形如 "北京-海淀区"，去掉区级后缀）
    if n > 2:
        city_raw = texts[2]
        job['city'] = city_raw.split('-')[0].strip() if '-' in city_raw else city_raw.strip()

    # 检查是否有角标（急聘/推荐），有的话会挤占字段位置
    sal_idx = 4
    badge_idx = 4
    if n > 4 and texts[4] in ('急聘', '推荐', '置顶', '广告'):
        badge_idx = 4
        sal_idx = 5

    # 薪资（通常在 sal_idx 位置）
    if n > sal_idx:
        sal_text = texts[sal_idx]
        job['salary_text'] = sal_text
        job['salary_min'], job['salary_max'] = parse_salary(sal_text)

    # 经验要求（通常在薪资后面一个位置）
    exp_idx = sal_idx + 1
    if n > exp_idx and ('年' in texts[exp_idx] or '应届' in texts[exp_idx] or '经验' in texts[exp_idx] or '以下' in texts[exp_idx]):
        job['experience_text'] = texts[exp_idx]
        job['exp_min'], job['exp_max'] = parse_experience(texts[exp_idx])

    # 学历要求（通常在经验后面一个位置）
    edu_idx = exp_idx + 1
    if n > edu_idx:
        edu_text = texts[edu_idx]
        if any(k in edu_text for k in ['博', '硕', '本', '大专', '不限']):
            job['education'] = parse_education(edu_text)

    # 公司名称（通过排除法找到真正的公司字段，跳过薪资/学历/经验等字段）
    company_idx = 7
    for ci in range(5, min(n, 9)):
        t = texts[ci]
        if (len(t) > 1
            and not any(k in t for k in ['【', '】', 'k', 'K', '¥'])
            and not re.match(r'^[\d.\-kK·×薪]+$', t)
            and not any(k in t for k in ['年', '经验', '应届', '以下', '以上'])
            and not any(k in t for k in ['博', '硕', '本', '大专', '不限'])):
            company_idx = ci
            break

    if n > company_idx:
        job['company'] = texts[company_idx]

    # 行业（公司后面一个位置）
    if n > company_idx + 1:
        job['industry'] = texts[company_idx + 1]

    # 剩余字段：融资阶段、公司规模、招聘者信息
    remain_start = company_idx + 2
    remain_texts = texts[remain_start:] if remain_start < n else []

    for rt in remain_texts:
        if rt in ('已上市', '融资未公开', 'A轮', 'B轮', 'C轮', 'D轮', '天使轮',
                  '战略融资', 'IPO上市', '未融资', '不需要融资'):
            job['funding_stage'] = rt
        elif '人' in rt:
            job['company_size'] = rt
        elif re.match(r'.*[男女][0-9A-Z一-鿿]', rt) or '女士' in rt or '先生' in rt or 'HR' in rt:
            # 招聘者姓名，跳过
            pass
        elif '前' in rt or '小时' in rt or '天' in rt or '在线' in rt or '活跃' in rt:
            # 活跃时间信息，跳过
            pass

    # 从标题中提取技能关键词
    skill_tags = extract_skills_from_title(job['job_title'])
    if skill_tags:
        job['skills'] = ';'.join(skill_tags)

    return job


def scrape_page(driver, url, keyword, city_raw=''):
    """爬取单页搜索结果，返回解析后的岗位列表"""
    try:
        driver.get(url)
        time.sleep(5)

        handle_verification(driver)

        # 等待岗位卡片加载完成（最多等 10 秒）
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.job-card-pc-container'))
            )
        except TimeoutException:
            pass

        # 滚动页面，触发懒加载的内容
        for _ in range(3):
            driver.execute_script('window.scrollBy(0, 800)')
            time.sleep(0.5)

        # 用 BeautifulSoup 解析页面 HTML
        soup = BeautifulSoup(driver.page_source, 'lxml')
        cards = extract_job_cards(soup)

        results = []
        for card in cards:
            job = parse_card(card, keyword, city_raw)
            # 只保留 AI 相关岗位
            if job and job['job_title'] and is_ai_related(job['job_title']):
                results.append(job)
        return results

    except Exception as e:
        log(f"  [FAIL] {e}")
        return []


def run():
    """主爬虫流程：遍历关键词 → 逐页爬取 → 去重合并"""
    log("=" * 60)
    log("猎聘网 AI 岗位爬虫启动")
    log(f"关键词 ({len(KEYWORDS)}): {', '.join(KEYWORDS)}")
    log(f"每个关键词翻 {MAX_PAGES_PER_KW} 页")
    log(f"提取城市信息时不依赖城市代码，直接从岗位卡片中解析")
    log("=" * 60)

    driver = create_driver()
    all_jobs = []
    seen = set()  # 用于去重，避免同一个岗位被多个关键词重复采集

    try:
        total_requests = len(KEYWORDS) * MAX_PAGES_PER_KW
        request_count = 0

        # 外层循环：遍历每个关键词
        for keyword in KEYWORDS:
            # 内层循环：翻页
            for page in range(MAX_PAGES_PER_KW):
                request_count += 1
                # 不指定 dq 参数 → 全国范围搜索
                url = (
                    f'https://www.liepin.com/zhaopin/'
                    f'?key={keyword}'
                    f'&currentPage={page}'
                    f'&pageSize={PAGE_SIZE}'
                )
                log(f"[{request_count}/{total_requests}] \"{keyword}\" 第 {page+1} 页")
                jobs = scrape_page(driver, url, keyword, '')

                # 去重逻辑：用 "岗位|公司|城市|薪资" 作为唯一标识
                added = 0
                for j in jobs:
                    if not is_ai_related(j['job_title']):
                        continue
                    key = f"{j['job_title']}|{j['company']}|{j['city']}|{j['salary_text']}"
                    if key not in seen:
                        seen.add(key)
                        all_jobs.append(j)
                        added += 1
                log(f"  → 新增 {added} 条 (累计 {len(all_jobs)})")

                # 随机延迟，避免请求过于频繁
                randsleep()

    except KeyboardInterrupt:
        log("\n用户中断爬取")
    except Exception as e:
        log(f"\n爬取出错: {e}")
    finally:
        driver.quit()  # 确保浏览器被关闭

    return all_jobs


def save_csv(jobs):
    """将爬取结果保存为 CSV 文件，并输出统计信息"""
    if not jobs:
        log("没有数据可保存")
        return

    # CSV 列定义
    fieldnames = [
        'job_title', 'city', 'salary_text', 'salary_min', 'salary_max',
        'education', 'experience_text', 'exp_min', 'exp_max',
        'company', 'industry', 'company_size', 'funding_stage',
        'employment_type', 'skills', 'keyword', 'city_raw', 'source', 'scrape_date'
    ]

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(job)

    log(f"\n[OK] 已保存 {len(jobs)} 条数据 -> {OUTPUT_FILE}")

    # ----- 统计概览 -----
    cities = set(j['city'] for j in jobs if j.get('city'))
    with_salary = sum(1 for j in jobs if j.get('salary_min'))
    with_edu = sum(1 for j in jobs if j.get('education'))
    log(f"  覆盖城市: {len(cities)} 个 - {', '.join(sorted(cities))}")
    log(f"  有薪资: {with_salary} 条 | 有学历: {with_edu} 条")

    # 按城市和关键词的分布统计
    from collections import Counter
    city_dist = Counter(j['city'] for j in jobs if j.get('city'))
    log(f"\n各城市岗位数:")
    for city, cnt in city_dist.most_common(10):
        avg_sal = None
        sals = [j['salary_min'] for j in jobs if j.get('city') == city and j.get('salary_min')]
        if sals:
            avg_sal = sum(sals) / len(sals)
        sal_str = f", 平均月薪 ¥{avg_sal:,.0f}" if avg_sal else ""
        log(f"  {city}: {cnt} 条{sal_str}")


def main():
    """入口函数：先爬取，再保存"""
    jobs = run()
    save_csv(jobs)


if __name__ == '__main__':
    main()

"""
Hugging Face 模型数据爬虫
从 HF-Mirror 爬取文本生成、图像分类、文生图三个类别的 Top 100 模型
"""
import urllib.request
import json
import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 全局配置 ============

# 输出目录：脚本所在文件夹
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# HF-Mirror API 地址
API_BASE = "https://hf-mirror.com/api/models"

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# 定义需要爬取的任务类别字典：pipeline_tag
TASKS = {
    "text-generation": "Text Generation",          # 文本生成
    "image-classification": "Image Classification", # 图像分类
    "text-to-image": "Text to Image"                # 文生图
}

# ============ 网络请求工具函数 ============

def fetch_json(url, timeout=60):
    """
    发送 HTTP GET 请求，返回解析后的 JSON 数据
    url: API 地址
    timeout: 超时时间（秒），默认 60 秒
    """
    req = urllib.request.Request(url, headers=HEADERS)
    # 发起请求 → 读取响应 → 解析 JSON
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def scrape_model_list(pipeline_tag, limit=100):
    """
    获取指定任务类别中下载量排名前 N 的模型列表
    pipeline_tag: 任务类别标识（如 "text-generation"）
    limit: 返回的模型数量上限
    返回: 模型列表（每个元素是一个字典）
    """
    # 按下载量降序排列，取前 limit 个模型
    url = f"{API_BASE}?pipeline_tag={pipeline_tag}&sort=downloads&direction=-1&limit={limit}&full=true"
    print(f"  Fetching list for {pipeline_tag}...")
    return fetch_json(url)

def scrape_model_detail(model_id):
    """
    获取单个模型的详细信息（主要用于提取参数量等额外元数据）
    model_id: 模型的唯一标识（如 "Qwen/Qwen2.5-7B"）
    返回: 模型详情字典，失败时返回 None
    """
    url = f"{API_BASE}/{model_id}"
    try:
        return fetch_json(url, timeout=30)
    except Exception as e:
        print(f"    Warning: Failed to fetch detail for {model_id}: {e}")
        return None

# ============ 字段提取函数 ============

def extract_arxiv(tags, card_data):
    """
    从 tags 或 cardData 中提取关联的学术论文信息
    优先从 tags 中找 "arxiv:XXXX.XXXX" 格式，其次从 cardData 中找 arxiv/paper 字段
    """
    # 方式一：从 tags 中提取 arxiv 标签
    arxiv_tags = [t for t in tags if t.startswith("arxiv:")]
    if arxiv_tags:
        return arxiv_tags[0].replace("arxiv:", "")

    # 方式二：从 cardData 中提取论文链接
    if card_data:
        for key in ['arxiv', 'paper']:
            if key in card_data:
                val = card_data[key]
                if isinstance(val, str) and val:
                    return val
                if isinstance(val, list) and val:
                    return val[0]
    return None

def extract_license(tags, card_data):
    """
    从 tags 或 cardData 中提取模型的开源许可证信息
    优先从 tags 中找 "license:xxx" 格式，其次从 cardData 的 license 字段获取
    """
    # 方式一：从 tags 中提取许可证标签
    license_tags = [t for t in tags if t.startswith("license:")]
    if license_tags:
        return license_tags[0].replace("license:", "")

    # 方式二：从 cardData 中提取许可证信息
    if card_data and 'license' in card_data:
        lic = card_data['license']
        if isinstance(lic, str) and lic:
            return lic
        if isinstance(lic, list) and lic:
            return lic[0]
    return None

def extract_frameworks(tags, library_name):
    """
    从 tags 和 library_name 中提取模型支持的深度学习框架列表
    返回排序后的框架名称列表（去重）
    """
    # 预定义已知的主流框架列表，用于匹配 tags
    known_frameworks = [
        "transformers", "pytorch", "tensorflow", "onnx", "gguf",
        "safetensors", "diffusers", "flax", "keras", "spacy",
        "openvino", "mlx", "rust", "peft", "bitsandbytes", "adapter",
        "nemo", "fastai", "allennlp", "trl", "text-generation-inference"
    ]
    frameworks = set()
    # 将库名称加入框架集合
    if library_name:
        frameworks.add(library_name)
    # 遍历 tags，匹配已知框架名
    for t in tags:
        t_lower = t.lower()
        if t_lower in known_frameworks:
            frameworks.add(t_lower)
    return sorted(frameworks)

def extract_parameters(model_detail):
    """
    从模型详细信息中提取参数数量
    数据来源为 safetensors 字段，可能以不同格式存储
    返回: 参数量（整数/浮点数），无法获取时返回 None
    """
    if not model_detail:
        return None

    safetensors = model_detail.get('safetensors')
    if not safetensors or not isinstance(safetensors, dict):
        return None

    # safetensors.total 可能是数字，也可能是嵌套字典
    total = safetensors.get('total') or safetensors.get('parameters', {})
    if isinstance(total, (int, float)):
        return total
    if isinstance(total, dict):
        # 如果是字典格式，对各模块参数量求和
        return sum(v for v in total.values() if isinstance(v, (int, float)))
    return None

def get_size_category(param_count):
    """
    将具体参数量映射为可读的规模等级标签
    用于后续的分组分析和可视化
    等级: <100M、100M-1B、1B-5B、…、100B+
    """
    if param_count is None:
        return "Unknown"

    params_b = param_count / 1e9  # 转换为十亿为单位
    if params_b >= 100:
        return "100B+"
    elif params_b >= 50:
        return "50B-100B"
    elif params_b >= 20:
        return "20B-50B"
    elif params_b >= 10:
        return "10B-20B"
    elif params_b >= 5:
        return "5B-10B"
    elif params_b >= 1:
        return "1B-5B"
    elif params_b >= 0.1:
        return "100M-1B"
    else:
        return "<100M"

# ============ 数据处理函数 ============

def process_models(task_name, models_list):
    """
    批量处理模型列表
    1. 并发拉取每个模型的详细信息（获取参数量等额外数据）
    2. 提取需要的字段，整理为结构化字典
    3. 返回处理后的结果列表
    task_name: 任务的中文可读名称
    models_list: 模型列表（来自 scrape_model_list）
    """
    results = []
    model_ids = [m['id'] for m in models_list]

    print(f"  Fetching details for {len(model_ids)} models (parallel)...")

    # 使用线程池并发拉取模型详情，将网络等待时间从串行变为并行
    details = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务到线程池
        fut_to_id = {executor.submit(scrape_model_detail, mid): mid for mid in model_ids}
        # 逐个处理返回结果
        for fut in as_completed(fut_to_id):
            mid = fut_to_id[fut]
            try:
                result = fut.result()
                if result:
                    details[mid] = result
            except Exception as e:
                print(f"    Error: {mid}: {e}")

    # 遍历模型列表，逐条提取并整理所需字段
    for m in models_list:
        model_id = m['id']
        tags = m.get('tags', [])
        card_data = m.get('cardData') or {}
        detail = details.get(model_id)

        # 提取参数量并映射为规模等级
        param_count = extract_parameters(detail)
        size_category = get_size_category(param_count)

        # 组装为结构化字典，字段名与后续分析的 DataFrame 列名对应
        result = {
            'model_id': model_id,                   # 模型唯一标识
            'downloads': m.get('downloads', 0),     # 下载量
            'likes': m.get('likes', 0),             # 点赞数
            'pipeline_tag': m.get('pipeline_tag', ''),  # 任务类别
            'task_name': task_name,                 # 任务可读名称
            'library_name': m.get('library_name', ''),  # 主要库/框架
            'frameworks': ', '.join(extract_frameworks(tags, m.get('library_name', ''))),  # 支持的框架列表（逗号分隔）
            'license': extract_license(tags, card_data),  # 开源许可证
            'arxiv': extract_arxiv(tags, card_data),      # 关联论文
            'param_count': param_count,                   # 参数量
            'size_category': size_category,               # 规模等级
            'created_at': m.get('createdAt', ''),         # 创建时间
            'last_modified': m.get('lastModified', ''),   # 最后修改时间
        }
        results.append(result)

    return results

# ============ 主函数 ============

def main():
    """
    主流程：
    1. 遍历每个任务类别，依次爬取 Top 100 模型
    2. 每个类别保存一个独立的 CSV 文件
    3. 最后将所有类别合并为一个总 CSV 文件
    """
    all_results = {}

    # 按任务类别依次爬取
    for pipeline_tag, task_name in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Scraping: {task_name} ({pipeline_tag})")
        print(f"{'='*60}")

        # 获取该类别下的模型列表
        models = scrape_model_list(pipeline_tag, limit=100)
        print(f"  Got {len(models)} models")

        # 批量处理（并发拉取详情、提取字段）
        results = process_models(task_name, models)
        all_results[pipeline_tag] = results

        # 保存该类别为独立的 CSV 文件
        csv_path = os.path.join(OUTPUT_DIR, f"hf_models_{pipeline_tag}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()    # 写入列名
                writer.writerows(results)  # 写入数据行
        print(f"  Saved {len(results)} models to {csv_path}")

        # 请求间隔 1 秒，避免触发频率限制
        time.sleep(1)

    # 合并所有类别的数据，保存为总 CSV
    all_rows = []
    for pipeline_tag in TASKS:
        all_rows.extend(all_results[pipeline_tag])

    combined_path = os.path.join(OUTPUT_DIR, "hf_models_all.csv")
    with open(combined_path, 'w', newline='', encoding='utf-8') as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\nSaved combined data ({len(all_rows)} rows) to {combined_path}")

# 脚本入口
if __name__ == "__main__":
    main()

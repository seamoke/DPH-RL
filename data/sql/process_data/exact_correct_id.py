import json
import os
from collections import defaultdict
import pandas as pd # 导入 pandas 库，用于处理 parquet 文件
import pdb

def extract_jsonl_data(file_path, id_groups=None):
    """
    从指定的 JSONL 文件中读取数据，并返回一个列表。
    新增功能：如果提供了 id_groups 参数，则会为 id == -1 的元素分配共享 ID。

    Args:
        file_path (str): JSONL 文件的完整路径。
        id_groups (int, optional): 每组共享一个 ID 的元素数量。
                                   例如，如果 id_groups=3，则前 3 个 id==-1 的元素
                                   将被分配 ID 0，接下来 3 个将被分配 ID 1，以此类推。
                                   默认为 None，不执行此逻辑。

    Returns:
        list: 包含文件中所有 item 的列表，其中部分 id 可能已被修改。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        return []
    
    if not os.access(file_path, os.R_OK):
        print(f"错误：文件 '{file_path}' 没有读取权限。")
        return []
        
    # 验证 id_groups 参数的有效性
    if id_groups is not None and not (isinstance(id_groups, int) and id_groups > 0):
        print(f"错误：id_groups 必须是一个正整数，但收到的值是：{id_groups}")
        return []

    data = []
    # 新增计数器，只在遇到 id == -1 的元素时递增
    neg_one_id_counter = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取每一行并解析 JSON 对象
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    
                    # --- 新增逻辑：处理共享 ID ---
                    # 仅在 id_groups 有效且当前元素 id 为 -1 时执行
                    # 使用 .get() 方法以避免因缺少 'id' 键而导致的 KeyError
                    #pdb.set_trace()
                    if id_groups is not None:
                        
                        # 计算共享 ID
                        shared_id = neg_one_id_counter // id_groups
                        item['id'] = shared_id
                        
                        # 为下一个 id == -1 的元素准备计数器
                        neg_one_id_counter += 1
                    # --- 逻辑结束 ---
                    
                    data.append(item)

                except json.JSONDecodeError as e:
                    print(f"警告：无法解析行，跳过。错误：{e}")
                    
    print(f"成功从 '{file_path}' 加载了 {len(data)} 条数据。")
    return data

def filter_and_save_correct_data():
    """
    读取两个指定的 JSONL 文件，根据 pass@8 条件筛选数据，并保存到新文件中。
    """
    # ================================================================= #
    #                         配置区域                                   #
    # ================================================================= #
    # 定义输入和输出文件路径
    source_file_path = './data/sql/bird_train.jsonl'
    results_file_path = './data/sql/llama3.1-8b/generate_data//0.jsonl'
    
    # .jsonl 文件输出路径
    output_correct_jsonl_path = './data/sql/llama3.1-8b/generate_data//correct_data.jsonl'
    output_wrong_jsonl_path = './data/sql/llama3.1-8b/generate_data//wrong_data.jsonl'

    # .parquet 文件输出路径
    output_correct_parquet_path = './data/sql/llama3.1-8b/generate_data//correct_data.parquet'
    output_wrong_parquet_path = './data/sql/llama3.1-8b/generate_data//wrong_data.parquet'

    # ================================================================= #
    #                         主逻辑开始                                 #
    # ================================================================= #
    
    # 1. 从 source_file_path 中加载所有数据，并以 ID 为键创建字典，方便快速查找
    print("--- 正在加载源数据... ---")
    source_data = extract_jsonl_data(source_file_path,1)
    
    if not source_data:
        print("源数据为空，程序终止。")
        return

    source_data_by_id = {item['id']: item for item in source_data}

    # 2. 从 results_file_path 中加载结果数据
    print("--- 正在加载结果数据... ---")
    results_data = extract_jsonl_data(results_file_path,4)
    if not results_data:
        print("结果数据为空，程序终止。")
        return
    pdb.set_trace()
    # 3. 筛选符合 pass@8 条件的数据
    print("--- 正在根据 pass@8 条件筛选数据... ---")
    correct_ids = set()
    completely_wrong_ids = set()
    partially_correct_wrong_ids = set()
    attempts_by_id = defaultdict(list)

    # 遍历结果数据，按 id 聚合 acc 列表
    for item in results_data:
        if 'id' in item and 'acc' in item:
            attempts_by_id[item['id']].append(item['acc'])

    # 遍历聚合后的结果，判断是否符合 pass@8 条件
    for item_id, acc_list in attempts_by_id.items():
        # 确保 acc 列表长度至少为 8
        if len(acc_list) >= 4:
            # 计算前 8 个尝试的成功次数
            pass8_successes = sum(acc_list[:4])
            # 判断是否满足 >= 6 的条件
            if pass8_successes >= 4:
                correct_ids.add(item_id)
            else:
                partially_correct_wrong_ids.add(item_id)
    
    print(f"找到了 {len(correct_ids)} 个符合 pass@8 条件的 ID。")
    print(f"找到了 {len(completely_wrong_ids)} 个8次尝试中全部错误的 ID。")
    print(f"找到了 {len(partially_correct_wrong_ids)} 个部分正确但未达到 pass@8 标准的 ID。")

    # 4. 根据筛选出的 ID 从源数据中提取完整项并分类
    correct_data = []
    wrong_data = [] # 仅包含部分正确但未达标的数据
    completely_wrong_data = [] # 仅包含全部错误的数据

    for item_id, item_data in source_data_by_id.items():
        if item_id in correct_ids:
            correct_data.append(item_data)
        elif item_id in completely_wrong_ids:
            completely_wrong_data.append(item_data)
        elif item_id in partially_correct_wrong_ids:
            wrong_data.append(item_data)

    print(f"共有 {len(correct_data)} 条正确数据。")
    print(f"共有 {len(wrong_data)} 条部分正确数据。")
    print(f"共有 {len(completely_wrong_data)} 条全部错误的数据，这些数据将被丢弃。")


    # 5. 将筛选出的数据保存到 .jsonl 文件 (只保存正确和部分正确的数据)
    print(f"--- 正在将数据保存为 .jsonl 文件... ---")
    try:
        with open(output_correct_jsonl_path, 'w', encoding='utf-8') as f:
            for item in correct_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"正确数据已成功保存到 '{output_correct_jsonl_path}'。")

        with open(output_wrong_jsonl_path, 'w', encoding='utf-8') as f:
            for item in wrong_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"部分正确数据已成功保存到 '{output_wrong_jsonl_path}'。")
    except IOError as e:
        print(f"错误：无法写入 .jsonl 文件。错误信息：{e}")

    # 6. 将筛选出的数据保存到 .parquet 文件 (只保存正确和部分正确的数据)
    print(f"--- 正在将数据保存为 .parquet 文件... ---")
    try:
        df_correct = pd.DataFrame(correct_data)
        df_correct.to_parquet(output_correct_parquet_path)
        print(f"正确数据已成功保存到 '{output_correct_parquet_path}'。")

        df_wrong = pd.DataFrame(wrong_data)
        df_wrong.to_parquet(output_wrong_parquet_path)
        print(f"部分正确数据已成功保存到 '{output_wrong_parquet_path}'。")
    except ImportError:
        print("错误：请安装 pandas 和 pyarrow 库以保存 .parquet 文件。命令：'pip install pandas pyarrow'")
    except Exception as e:
        print(f"错误：无法写入 .parquet 文件。错误信息：{e}")


# --- 程序入口 ---
if __name__ == '__main__':
    filter_and_save_correct_data()

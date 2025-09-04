
import pdb
import os, re
import requests
import random
import json, time
import asyncio
import asyncio
import copy
import aiohttp
from typing import Literal
from tqdm import tqdm
import threading

def program_extract(text: str,
                    program: str = "sql",
                    mode: Literal["first", "last", "all"] = "last") -> str:

    program_pattern = rf"```{program}[ \t]*[\r\n]+(.*?)[\r\n]+[ \t]*```"
    program_re = re.compile(program_pattern, re.DOTALL | re.IGNORECASE)
    matches = program_re.findall(text)
    if matches:
        if mode == "first":
            return matches[0].strip()
        elif mode == "last":
            return matches[-1].strip()
        elif mode == "all":
            return "\n\n".join(matches)
    else:
        print('INVALID')
        #print(f'INVALID:{text}')
        return "INVALID!"

def extract_tool_call_content(solution_str):
    """
    Extracts the content within the last <tool_call>...</tool_call> tags
    from a given string.
    """
    all_matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', solution_str, re.DOTALL))
    if not all_matches:
        return None

    last_match = all_matches[-1]
    # Group 1 captures the content inside the tags due to the parentheses in the regex
    return last_match.group(1)

def format_compute_score(solution_str):
    try:
        tool_call_content = extract_tool_call_content(solution_str)

        if tool_call_content is None:
            return 0.0, solution_str

        # Now, parse the JSON content
        # Assuming the content within tool_call is valid JSON,
        # you might need to handle cases where it's not strictly JSON,
        # or has extra characters like the original example's ````json{` and `}```
        # For the example you provided, the content is already pure JSON.
        solution_json = json.loads(tool_call_content)
        predicted_shares = solution_json['arguments']['sql']

        # If either value is None, return 0
        if predicted_shares is not None:
            return 0.2, solution_str
        return 0.0, solution_str

    except Exception as e:
        print(f"An error occurred: {e}") # Added for debugging
        return 0.0, solution_str
    
def read_files_to_strings(directory_path):
    """
    Reads all files in a given directory and stores their content as strings.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        dict: A dictionary where keys are filenames and values are the file contents as strings.
              Returns an empty dictionary if the directory does not exist or is empty.
    """
    file_contents = []
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return ["0.0.0.0"]

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip() # .strip() removes leading/trailing whitespace, including newlines
                    file_contents.append(content)
            except Exception as e:
                print(f"Error reading file '{filename}': {e}")
    return file_contents

ip_dir = os.environ.get('EXPERIMENT_NAME','servers')
# --- Usage Example ---
directory_to_read = f"./ip_tmp/{ip_dir}"
print(ip_dir)
all_server_ips = read_files_to_strings(directory_to_read)
print("All server IPs:", all_server_ips)
unless_temp = '```sql\n-- Your SQL query\n```\n'

MAX_CONCURRENT_REQUESTS = 180
traffic_limiter = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
def score(solution_str, ground_truth, only_val=False):
    """
    同步版本的 score 函数，用于评估解决方案。

    Args:
        solution_str (str): 待评估的解决方案字符串。
        ground_truth (any): 评估的参照物。
        only_val (bool): 如果为True，只返回分数。

    Returns:
        tuple or dict: 如果 only_val 为 False，返回 (comp, res)；
                       如果 only_val 为 True，返回 {'score': comp, 'acc': acc}。
    """
    format_score = 0
    if only_val:
        # 假设 format_compute_score 不会崩溃
        format_score, _ = format_compute_score(solution_str.replace(unless_temp, ''))
    if '/' not in ground_truth['db_id']:
        folder_name = ground_truth['db_id'].replace('.sqlite', '')
        # 使用 os.path.join 拼接成 '文件夹名/文件名.sqlite'
        ground_truth['db_id'] = os.path.join(folder_name, ground_truth['db_id'])
    trajs = {
        'gen_responses': [solution_str.replace(unless_temp, '')],
        'extra': ground_truth
    }
    comp = 0
    res = ""
    acc = 0

    # 使用 with 语句来确保信号量被正确释放
    with traffic_limiter:
        # 以下是你的原始代码，现在被信号量包裹
        try:
            headers = {'Content-Type': 'application/json;charset=UTF-8'}
            max_retries = 3
            response_json = ''

            for attempt in range(max_retries):
                # 在同步环境中，使用 time.sleep() 是正常的。
                copy_server_ips = copy.deepcopy(all_server_ips)
                random.shuffle(copy_server_ips)
                choose_ip = copy_server_ips[0]

                try:
                    # 使用 requests 发起同步请求
                    response = requests.put(
                        f'http://{choose_ip}:5001/api',
                        json=trajs,
                        headers=headers,
                        # requests 的超时设置
                        timeout=3
                    )
                    response.raise_for_status()  # 检查响应状态
                    response_json = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    print(f"请求在第 {attempt + 1} 次尝试时失败，正在重试... 错误: {e}")
                except Exception as e:
                    # 捕获其他可能的异常
                    print(f"发生未知错误: {e}")

            # 检查 response_json 是否为字典且包含所需键
            if isinstance(response_json, dict) and 'Scorer1' in response_json and isinstance(response_json['Scorer1'], list):
                if len(response_json['Scorer1']) > 0 and isinstance(response_json['Scorer1'][0], list):
                    # 列表推导式
                    correct = [score[0] for score in response_json['Scorer1'] if isinstance(score, list) and len(score) > 0]
                    if correct:
                        comp = correct[0]
                        # 安全地访问多层嵌套数据
                        if len(response_json['Scorer1'][0]) > 1 and len(response_json['Scorer1'][0][1]) > 0 and len(response_json['Scorer1'][0][1][0]) > 0:
                            res = response_json['Scorer1'][0][1][0][-1]
            eps = 0.0001
            acc = 1.0 if abs(comp-1.0) < eps else 0
            comp += format_score

            if only_val:
                return {
                    "score": comp,
                    "acc": acc
                }
            return comp, res

        except Exception as e:
            # 捕获函数执行过程中所有未处理的异常
            print(f"在 score 函数执行过程中发生致命错误: {e}")
            # 返回默认值，防止程序崩溃
            if only_val:
                return {
                    "score": 0,
                    "acc": 0
                }
            return 0, ""

async def score_async(solution_str, ground_truth, only_val=False):
    """
    异步版本的 score 函数，用于评估解决方案。

    Args:
        solution_str (str): 待评估的解决方案字符串。
        ground_truth (any): 评估的参照物。
        only_val (bool): 如果为True，只返回分数。

    Returns:
        tuple or dict: 如果 only_val 为 False，返回 (comp, res)；
                       如果 only_val 为 True，返回 {'score': comp, 'acc': acc}。
    """
    format_score = 0
    if only_val:
        format_score, _ = format_compute_score(solution_str.replace(unless_temp, ''))

    trajs = {
        'gen_responses': [solution_str.replace(unless_temp, '')],
        'extra': ground_truth
    }
    res = ""
    comp = 0
    
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    max_retries = 3
    response_json = ''

    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            # 异步环境中不建议使用 time.sleep()，因为它会阻塞事件循环。
            # 替代方案是 asyncio.sleep()。
            await asyncio.sleep(1)
            copy_server_ips = copy.deepcopy(all_server_ips) 
            random.shuffle(copy_server_ips)
            choose_ip = copy_server_ips[0]

            try:
                # 使用 aiohttp 发起异步请求
                async with session.put(
                    f'http://{choose_ip}:5001/api', 
                    json=trajs, 
                    headers=headers, 
                    # aiohttp 的超时设置
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    response.raise_for_status()  # 检查响应状态
                    response_json = await response.json()
                    break
            except aiohttp.ClientError as e:
                print(f"请求在第 {attempt + 1} 次尝试时失败，正在重试... 错误: {e}")
            except asyncio.TimeoutError:
                print(f"请求在第 {attempt + 1} 次尝试时超时。正在重试...")

    if 'Scorer1' in response_json and isinstance(response_json['Scorer1'], list):
        if len(response_json['Scorer1']) > 0 and isinstance(response_json['Scorer1'][0], list):
            # 列表推导式
            correct = [score[0] for score in response_json['Scorer1'] if isinstance(score, list) and len(score) > 0]
            if correct:
                comp = correct[0]
                res = response_json['Scorer1'][0][1][0][-1]
    pdb.set_trace()
    acc = comp
    comp += format_score

    if only_val:
        return {
            "score": comp,
            "acc": acc
        }
    print(f"Type of res before return: {type(res)}")
    return comp, res
# def score(solution_str,ground_truth,only_val=False):
#     # if only_val: 
#     #     return 1
#     # return 1,[[1],[2],[3]]
#     queries = [solution_str]

#     trajs = ground_truth

#     gold_sql = trajs['gold_sql']

#     db_id = os.path.join('/cache/', trajs['db_id'])


#     cmp_method = trajs.get('cmp_method', 'spider')

#     gold_answer = excecutor.exec_sql(db_id, [gold_sql],

#     keep_distinct=True,

#     do_post_process=False)

#     if gold_answer[0][0] != 'result':

#         print(f"db_id:{db_id} gold_sql: {gold_sql} is not executable. result:{gold_answer}")
#         return 0, 'error'


#     exec_res = []

#     unless_temp = '```sql\n-- Your SQL query\n```\n'

#     for query in queries:

#         res = excecutor.exec_sql(db_id, [query.replace(unless_temp, '')], keep_distinct=True)

#         exec_res.append(res)


#     comp_res = []

#     for res in exec_res:

#         if res[0][0] != 'result':
#             comp_res.append(False)
#         #print(f"execution error:{res}", flush=True)

#         else:
#             if cmp_method == "spider":
#                 order_matters = "orderby" in re.sub(r"\s+", gold_sql.lower(), "")
#                 comp = eq_func(res[0][-1], gold_answer[0][-1], order_matters)
#                 comp_res.append(comp)
#             else:
#                 comp = set(res[0][-1]) == set(gold_answer[0][-1])
#                 comp_res.append(comp)

#     #print(f'generation:{res[0][-1][0:3]}\ngold:{gold_answer[0][-1][0:3]}\ncomp:{comp}')

#     comp_res = [1.0 if res else 0.0 for res in comp_res]

#     assert len(comp_res) == len(queries)

#     if only_val:
#         return comp_res[0]
#     return comp_res[0], exec_res[0]
trajs = {"cmp_method": "bird", "db_id": "retails.sqlite", "gold_sql": "SELECT COUNT(T3.s_name) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey INNER JOIN nation AS T4 ON T3.s_nationkey = T4.n_nationkey WHERE T1.p_name = 'hot spring dodger dim light' AND T4.n_name = 'VIETNAM'"}
async def calc_reward():
    return await score(
        "```sql\nSELECT COUNT(T3.s_name) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey INNER JOIN nation AS T4 ON T3.s_nationkey = T4.n_nationkey WHERE T1.p_name = 'hot spring dodger dim light' AND T4.n_name = 'VIETNAM'\n```",
        trajs,
    )

if __name__ == "__main__":
    pass
    
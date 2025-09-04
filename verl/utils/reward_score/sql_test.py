from sql import program_extract, score, unless_temp
print("This is a utility module for SQL tool scoring and execution.")
    # Example usage
    # import json
import concurrent.futures
import random
import threading,json
from tqdm import tqdm
file_path = 'your_path/data/sql/llama3.1-8b/generate_data/0.jsonl'
data = []
ref_path = 'your_path/data/sql/bird_dev.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 每一行都是一个独立的 JSON 对象
        data.append(json.loads(line.strip()))
ref_data = []
with open(ref_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 每一行都是一个独立的 JSON 对象
        ref_data.append(json.loads(line.strip()))
querys = '''1'''

#print(program_extract(querys, program='sql', mode='last'))
#score_result = score(querys, trajs, only_val=True)
lock = threading.Lock()
tot_score = 0
data_num = 10000
def process_item(item_ref_tuple):
    """
    处理单个数据项并返回其得分。
    item_ref_tuple 是一个包含 (item, ref) 的元组。
    """
    item, ref = item_ref_tuple
    try:
        query = item['output']
        # 假设 program_extract 是一个已定义的函数
        query = program_extract(query.replace(unless_temp, ''), program='sql', mode='last')
        query = f"```sql\n{query}\n```"
        
        acc = item['acc']
        trajs = ref['reward_model']['ground_truth']
        # 假设 score 是一个已定义的函数
        score_result = score(query, trajs, only_val=True)
        return score_result
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return {
                    "score": 0,
                    "acc": 0
                }

items = data[0:data_num]
refs = ref_data[0:data_num]
# 将 item 和 ref 打包成元组，以便 map 函数处理
data_to_process = zip(items, refs)


# with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
#     # 使用 map 方法提交任务，它会返回一个迭代器，结果按提交顺序返回
#     results = executor.map(process_item, data_to_process)
    
#     # 遍历结果并处理
#     for i, score_result in enumerate(tqdm(results, total=len(items))):
#         if score_result:
#             #print(f"{i+1}:{score_result}\n")
#             with lock:
#                 tot_score += score_result['acc']
for i,(item,ref) in enumerate(zip(data[0:data_num],ref_data[0:data_num])):
    query = item['output']
    query = program_extract(query.replace(unless_temp,''), program='sql', mode='last')
    query = f"```sql\n{query}\n```"
    #pdb.set_trace()
    acc = item['acc']
    trajs = ref['reward_model']['ground_truth']
    score_result = score(query, trajs, only_val=True)
    print(f"{i+1}:{score_result}\n")
    if acc != score_result['acc']:
        print("Acc not match!")
        print("Query:", query)
        print("Trajs:", trajs)
        print("Expected acc:", acc)
        print("Got acc:", score_result['acc'])
        import pdb;pdb.set_trace()
    tot_score += score_result['acc']

print("Score result:", tot_score/min(data_num,len(items)))
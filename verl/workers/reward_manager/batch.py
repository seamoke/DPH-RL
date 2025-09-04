# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MODIFICATION 1: We now import from `concurrent.futures` instead of `asyncio`.
import concurrent.futures
from collections import defaultdict
import torch
from verl import DataProto
from verl.workers.reward_manager import register


@register("batch")
class BatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.
    This version uses a ThreadPoolExecutor for concurrency within a synchronous interface.
    """

    # MODIFICATION 2: Added `max_workers` to control the size of the thread pool.
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", max_workers=32, **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.max_workers = max_workers
        
        # MODIFICATION 3: Create a reusable thread pool executor.
        # This is more efficient than creating a new one for every call.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

    def verify(self, data):
        # ... (verify method remains unchanged)
        # ... (code omitted for brevity)
        pass

    # MODIFICATION 4: `__call__` is a standard synchronous method (`def`).
    def __call__(self, data: DataProto, return_dict=False):
        """
        This is a synchronous method that internally uses a thread pool for concurrency.
        """
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        print(f"Evaling {len(data)} data")

        tasks_data = []
        # ... (data preparation loop remains the same, code omitted for brevity)
        for i, data_item in enumerate(data):
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            tasks_data.append({
                "index": i, "solution_str": self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
                "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                "data_source": data_item.non_tensor_batch[self.reward_fn_key],
                "extra_info": data_item.non_tensor_batch.get("extra_info", None),
                "prompt_str": self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                "valid_response_length": valid_response_length,
            })
        
        # MODIFICATION 5: Use the thread pool to submit tasks.
        # This is the new concurrent execution logic.
        futures = []
        # A dictionary to map future objects back to their original index
        future_to_index = {}
        for i, task_info in enumerate(tasks_data):
            # `submit` schedules the function to be executed and returns a `Future` object.
            future = self.executor.submit(
                self.compute_score,
                data_source=task_info["data_source"],
                solution_str=task_info["solution_str"],
                ground_truth=task_info["ground_truth"],
                **self.reward_kwargs
            )
            futures.append(future)
            future_to_index[future] = i

        # Collect results and handle exceptions
        results = [None] * len(tasks_data)
        for future in concurrent.futures.as_completed(futures):
            index = future_to_index[future]
            try:
                # `future.result()` waits for the task to complete and returns its result.
                # If the task raised an exception, `result()` will re-raise it here.
                results[index] = future.result()
            except Exception as e:
                # Store the exception itself to be handled in the processing loop.
                results[index] = e

        # ... (result processing loop remains almost the same)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i, result in enumerate(results):
            task_info = tasks_data[i]
            if isinstance(result, Exception):
                print(f'[ERROR] Item {i} failed with exception: {result}')
                reward = 0.0
                score_dict = None
            else:
                score = result
                score_dict = None
                if isinstance(score, dict):
                    reward = score.get("score", 0.0)
                    score_dict = score
                else:
                    reward = score
            
            valid_response_length = task_info["valid_response_length"]
            if score_dict:
                for key, value in score_dict.items():
                    reward_extra_info[key].append(value)
            
            reward_tensor[i, valid_response_length - 1] = reward

            data_source = task_info["data_source"]
            # ... (printing logic remains the same)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # ... (print statements)
                
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def __del__(self):
        # Optional: ensure the thread pool is shut down when the object is garbage collected.
        self.executor.shutdown(wait=False)
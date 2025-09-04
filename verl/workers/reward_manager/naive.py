# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # reward_extra_info needs to be thread-safe or aggregated after parallel execution
        # For simplicity, we'll collect results and then populate
        results = [None] * len(data)
        
        print(f"Evaling {len(data)} data")
        timeout = 180
        # Determine the number of workers. You can adjust this based on your system's capabilities.
        # A common choice is the number of CPU cores or slightly more.
        max_workers = 64 # Example: use 4 threads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self._process_single_item, data_item, i): i for i, data_item in enumerate(data)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    # 使用 future.result() 获取结果，并设置超时
                    result = future.result(timeout=timeout)
                    results[index] = result
                    current_reward = result['reward']
                    print(f"✅ Task {index} finished with result: {current_reward}")
                except TimeoutError:
                    print(f"❌ Error: Task {index} timed out after {timeout} seconds.")
                    results[index] = 0.0
                except Exception as exc:
                    print(f"❌ Task {index} generated an exception: {exc}")
                    results[index] = 0.0
                    
        
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i, result in enumerate(results):
            if result is None:
                continue # Skip if an error occurred for this item

            reward = result["reward"]
            score_dict = result["score_dict"]
            valid_response_length = result["valid_response_length"]
            prompt_str = result["prompt_str"]
            response_str = result["response_str"]
            ground_truth = result["ground_truth"]
            data_source = result["data_source"]

            if score_dict:
                for key, value in score_dict.items():
                    reward_extra_info[key].append(value)
            
            reward_tensor[i, valid_response_length - 1] = reward

            # The printing logic needs to be outside the parallel part
            # to avoid race conditions or interleaved output
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if score_dict:
                    for key, value in score_dict.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", reward) # If it's not a dict, 'reward' is the score

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _process_single_item(self, data_item, index):
        """Helper function to process a single data item."""
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        if 'type' in data_item.batch and data_item.batch['type'] == 2:
            score = 1
        score = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        score_dict = None
        if isinstance(score, dict):
            reward = score["score"]
            score_dict = score
        else:
            reward = score
        
        return {
            "index": index,
            "reward": reward,
            "score_dict": score_dict,
            "valid_response_length": valid_response_length,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source,
        }
    # def __call__(self, data: DataProto, return_dict=False):
    #     """We will expand this function gradually based on the available datasets"""

    #     # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
    #     if "rm_scores" in data.batch.keys():
    #         if return_dict:
    #             return {"reward_tensor": data.batch["rm_scores"]}
    #         else:
    #             return data.batch["rm_scores"]

    #     reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    #     reward_extra_info = defaultdict(list)

    #     already_print_data_sources = {}

    #     for i in range(len(data)):
    #         data_item = data[i]  # DataProtoItem

    #         prompt_ids = data_item.batch["prompts"]

    #         prompt_length = prompt_ids.shape[-1]

    #         valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    #         response_ids = data_item.batch["responses"]
    #         valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    #         valid_response_ids = response_ids[:valid_response_length]

    #         # decode
    #         prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    #         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    #         ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    #         data_source = data_item.non_tensor_batch[self.reward_fn_key]
    #         extra_info = data_item.non_tensor_batch.get("extra_info", None)

    #         score = self.compute_score(
    #             data_source=data_source,
    #             solution_str=response_str,
    #             ground_truth=ground_truth,
    #             extra_info=extra_info,
    #         )

    #         if isinstance(score, dict):
    #             reward = score["score"]
    #             # Store the information including original reward
    #             for key, value in score.items():
    #                 reward_extra_info[key].append(value)
    #         else:
    #             reward = score

    #         reward_tensor[i, valid_response_length - 1] = reward

    #         if data_source not in already_print_data_sources:
    #             already_print_data_sources[data_source] = 0

    #         if already_print_data_sources[data_source] < self.num_examine:
    #             already_print_data_sources[data_source] += 1
    #             print("[prompt]", prompt_str)
    #             print("[response]", response_str)
    #             print("[ground_truth]", ground_truth)
    #             if isinstance(score, dict):
    #                 for key, value in score.items():
    #                     print(f"[{key}]", value)
    #             else:
    #                 print("[score]", score)

    #     if return_dict:
    #         return {
    #             "reward_tensor": reward_tensor,
    #             "reward_extra_info": reward_extra_info,
    #         }
    #     else:
    #         return reward_tensor

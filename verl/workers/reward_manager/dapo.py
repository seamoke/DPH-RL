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

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional
import torch
from transformers import PreTrainedTokenizer
from tqdm.asyncio import tqdm_asyncio
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import psutil
from collections import defaultdict
import time
import sys


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout):
    print(f"[DEBUG] Starting process_row_with_timeout for task: {task}", end="", flush=True)  # 调试信息
    # print("")
    # sys.stdout.flush()
    loop = asyncio.get_running_loop()
    start_time = time.time()  # Start time for debugging
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(evaluation_func, task, completion, reference, task_extra_info)
            ),
            timeout=timeout
        )
        # sys.stdout.flush()
        # print(f"[DEBUG] process_row_with_timeout completed in {time.time() - start_time:.2f}s for task: {task}")  # 调试信息
        return result
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout after {timeout}s: {completion[:10]}")
        return None
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:10]}")
        return None

async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes, timeout):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    print(f"[DEBUG] Starting parallel_evaluate_continual_async with {num_processes} processes")
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        try:
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout)
                for c, r, t, ei in zip(completions, references, tasks, extra_info)
            ]
            print(f"[DEBUG] Created {len(tasks_async)} async tasks")
            print(f"verifying {len(tasks_async)} tasks...")
            # results = await asyncio.gather(*tasks_async, return_exceptions=False)
            print("[DEBUG] Starting asyncio.gather with tqdm progress bar")
            results = await tqdm_asyncio.gather(
                *tasks_async, 
                desc="Processing tasks", 
                unit="task",
                # return_exceptions=False,
            )
            # print(f"[DEBUG] asyncio.gather completed in {time.time() - gather_start:.2f}s")
            print("[Success] All tasks gathered.")
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            print("[Shutdown] Cleaning up remaining subprocesses...")
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Format results
    formatted = []
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            formatted.append({
                "score": 0.,
                "format_score": 0.,
                "acc": False,
                "extracted_gt": reference,
                # "extracted_pred": None,
            })
        elif isinstance(result, dict):
            formatted.append(result)
        else:
            formatted.append(result[0])
    return formatted

def run_reward_scoring(compute_score_func, completions, references, tasks, extra_info, num_processes=64, timeout=300.):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(parallel_compute_score_async(
            compute_score_func, completions, references, tasks, extra_info, 
            num_processes, timeout
        ))
    finally:
        loop.close()


class DAPORewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # batched scoring
        response_ids = data.batch['responses']
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = data.non_tensor_batch.get('extra_info', None)

        assert len(responses_str) == len(ground_truths) == len(data_sources)
        try:
            results = run_reward_scoring(
                self.compute_score,
                completions=responses_str,
                references=ground_truths,
                tasks=data_sources,
                extra_info=extra_infos,
                num_processes=64,
                timeout=300.,
            )
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            results = [{
                "score": 0.,
                "format_score": 0.,
                "acc": False,
                "extracted_gt": gt,
                # "extracted_pred": None,
            } for gt in ground_truths]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            results = [{
                "score": 0.,
                "format_score": 0.,
                "acc": False,
                "extracted_gt": gt,
                # "extracted_pred": None,
            } for gt in ground_truths]
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            result = results[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            assert not isinstance(score, list), f"{score=} is list"
            reward = score


            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
        
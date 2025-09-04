# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import pdb 
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import sql

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

return_sql_template = '''
The return of the above SQL is:
EXECUTION RESULT (first 3 rows):
{sql_result}

If you think this is not correct. Refine this sql. Please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```
If you think this is correct. Just output "Final Answer.".
'''
class SqlTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "sql": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        sql = parameters.get("sql", "")

        if '```sql' not in sql:
            sql = f"```sql\n{sql}\n```"
        self._instance_dict[instance_id]["sql"] = sql
        reward, exec_res = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        print(f"Type of reward before return: {type(reward)}")
        print(f"Type of reward before return: {type(exec_res)}")
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward

        return return_sql_template.format(sql_result=str(exec_res[0:3])[0:256]), tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return sql.score(
            self._instance_dict[instance_id]["sql"],
            self._instance_dict[instance_id]["ground_truth"],
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

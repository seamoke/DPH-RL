<div align="center">

# The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward


<p align="center">
  <!-- Badges: replace placeholders with real links -->
  <a href="https://arxiv.org/abs/2509.07430"><img src="https://img.shields.io/badge/arXiv-2509.07430-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/EIT-NLP/LLaSO"><img src="https://img.shields.io/github/stars/seamoke/DPH-RL?style=social" alt="GitHub Stars"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-9cf.svg" alt="Cite"></a>
</p>


</div>

<p align="center">
  If you find DPH-RL useful, please ⭐ star this repo!
</p>

## 🔍 What is DPH-RL?
DPH-RL is an RL algorithm improved upon GRPO. It maintains policy diversity by pre-calculating the f-divergence from reference policy samples, which removes the need to load a reference model.

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./figures/method.png" width="" alt="Method"><br>
        <i>Method</i>
      </td>
    </tr>
  </table>
</p>
<p align='center'> <i>
</i></p>

### Result
Math and SQL generation experiments show that DPH-RL both improves in-domain Pass@1 and Pass@k scores and effectively prevents catastrophic forgetting on out-of-domain tasks.
<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./figures/image-1.png" width="450" alt="Method"><br>
        <i>In-Domain</i>
      </td>
    </tr>
  </table>
</p>
<p align='center'> <i>
</i></p>


<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./figures/result.png" width="" alt="Method"><br>
        <i>OOD and Keep</i>
      </td>
    </tr>
  </table>
</p>
<p align='center'> <i>
</i></p>
# ✨Getting started


## Data Prepared
This repo is forked from [verl](https://github.com/volcengine/verl). 
For SQL tasks, you need to load the databases. Please download the bird train and dev databases from [bird bench](https://bird-bench.github.io/), spider dev databases from [spider](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing). Then copy the databases to your local dir `/cache/`. Your /cache/ directory should look like the image below.
![alt text](./figures/image.png)

Our data is split into three parts:

### 1. Original Training Data
* `data/sql/bird_train.parquet`
---
### 2. Test Sets
* `data/sql/spider_dev.parquet`
* `data/sql/bird_dev.parquet`
---
### 3. Pre-Sampling Stage Data
This is our extra collected training data, referred to as $\mathcal{D}_{\text{exp}}$ and $\mathcal{D}_{\text{pef}}$ in the paper. You can find it in the `data/sql/llama3.1-8b/` directory.

## Installation
You can install dependencies by running the following commands:
```
pip install requirements.txt
```

## Evaluation

We launch a server for evaluation using the following code. 
```
cd your_path
python rl/scorer/scorer_server_without_ray.py -c ./rl/scorer/sql.yaml
```
Then, evaluation is performed via the IP address and port. For an implementation reference, please see the code in `verl/utils/reward_score/sql.py`.

## Training
Notes: Please remember set your `SWANLAB_API_KEY` or `WANDB_API_KEY`. All of our scripts are based on multi-machine deployment. We've mounted the evaluation server on non-Rank 0 machines to reduce cluster load. If you only have a single machine, please launch the evaluation server separately.
### Pre-Sampling Stage
#### For the llama-sql experiment, you can skip this step and directly use the data in `data/llama3.1-8b/`.
For DPH-RL, you need to split a complete dataset into two sub-datasets by performing a correct-ness check k times. This requires the following steps:

```
bash scripts/llama/offline_sampling.sh sampling
```
This script samples the training data eight times and saves the data by default to `$PROJECT_DIR/data/sql/llama3.1-8b/generate_data/0.jsonl`.

Next, use 
```
python ./data/sql/process_data/exact_correct_id.py
```
it split the data into `data/sql/llama3.1-8b/train_wrong.parquet` and `data/sql/llama3.1-8b/train_correct.parquet`.

To facilitate further exploration, we first generated data in the train format from `data/sql/llama3.1-8b/train_correct.parquet`. We sampled each data point only once, and saved the correct data to `$PROJECT_DIR/data/sql/llama3.1-8b/8b_llama3.1_all_right.pt`. Please use 
```
bash scripts/llama/get_correct_data_tensor.sh
```
Now, you can directly load this .pt file for model training. The `actor_rollout_ref.actor.generate_sft` parameter can be used to determine whether to sample SFT data.

### DPH-RL
You can implement different methods by directly calling the corresponding scripts in `scripts/llama`. The **`sft_loss_mode`** and **`sft_loss_coeff`** parameters are used to select the specific method and adjust hyperparameters.

The following table outlines the key considerations:
default set `use_kl_loss=False`
| `sft_loss_mode` | Description | Additional Settings | `sft_loss_coeff` |
| :--- | :--- | :--- | :--- |
| **forward** |**Forward KL** |None | 0.01~0.05|
| **js** | **The JS definition**|  `use_kl_loss=True` | 0.05~0.2|
| **js_low_var** | **The JS Generator**|  `use_kl_loss=False` | 0.05~0.2|
| **reverse_kl** | **Reverse KL**|  `data.sft_files=None`<br> `data.sft_pt=None` | 0.01~0.05|
| **alpha** | **$\alpha$ divergence** |  `data.sft_files=None`<br> `data.sft_pt=None` | 0.01~0.05|

<a id="citation"></a>
## 📑 Cite

If this paper can help you, please cite it:

```bibtex
@misc{li2025choicedivergenceneglectedkey,
      title={The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward}, 
      author={Long Li and Jiaran Hao and Jason Klein Liu and Zhijian Zhou and Xiaoyu Tan and Wei Chu and Zhe Wang and Shirui Pan and Chao Qu and Yuan Qi},
      year={2025},
      eprint={2509.07430},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.07430}, 
}
```
#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project

set -x
ulimit -n 65535
# cp -r /home/ma-user/work/jiaran/MLM-master/Megatron-LM/cache/* /cache

cd your_path
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR//examples/sglang_multiturn/config"
#pip install editdistance
# export NCCL_SOCKET_IFNAME=ens2f5
# export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_DEBUG=INFO

export HYDRA_FULL_ERROR=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export HOST_IP=0.0.0.0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_HOST_IP=0.0.0.0
#export TORCH_USE_CUDA_DSA=1
#pip install pylatexenc vertexai sentence_transformers

export HF_ENDPOINT=https://hf-mirror.com

nnode=1
export SWANLAB_API_KEY='your_swanlab_token'
export MODEL_PATH=your_model_path/Qwen3-4B-Base
export MODEL_PATH=your_model_path/Qwen2.5-0.5B-Instruct

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ma-user/anaconda3/envs/logic/lib/python3.10/site-packages/nvidia/cudnn/lib
maxlen=${maxlen:-"4096"}
nsample=${nsample:-"1"}

PROJECT_NAME=sql_multi_turn
EXPERIMENT_NAME=7b_${nsample}-${nnode}node
export VLLM_USE_V1=1
export PYTHONPATH="$PROJECT_DIR/:$PYTHONPATH"
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='sql_single_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    +data.dataloader_num_workers=0 \
    data.val_batch_size=1000 \
    data.max_prompt_length=16000 \
    data.max_response_length=${maxlen} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.generate_sft=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.sft_loss_mode=alpha \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${nsample} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='sql_async_rl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    +trainer.rollout_data_dir=$PROJECT_DIR/verl/data/sql/rollout_small \
    +trainer.validation_data_dir=$PROJECT_DIR/verl/data/sql/validation_small \
    trainer.save_freq=48 \
    trainer.test_freq=24 \
    trainer.default_local_dir=$PROJECT_DIR/rl_models \
    data.train_files=$PROJECT_DIR//data/sql/llama3.1-8b/train_wrong.parquet \
    data.sft_files=$PROJECT_DIR//data/sql/llama3.1-8b/train_correct.parquet \
    +data.sft_pt='$PROJECT_DIR/data/sql/llama3.1-8b/8b_llama3.1_all_right.pt' \
    data.sft_batch_size=512 \
    data.val_files=$PROJECT_DIR//data/sql/bird_dev.parquet \
    trainer.total_epochs=5
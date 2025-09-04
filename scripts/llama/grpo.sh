#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project

set -x
ulimit -n 65535

find_interface() {
  local ip_output=$(ip addr show | head -n 10) # Limit to first 10 lines
  local selected_interface=""

  # Debug output (can be removed in final version)
  # echo "--- First 10 lines of ip addr show output: ---"
  # echo "$ip_output"
  # echo "--- End of ip addr show output ---"

  while IFS= read -r line; do
    # Debug output (can be removed in final version)
    # echo "Processing line: $line"

    if [[ "$line" =~ ^[0-9]+:\ ([^:]+):\ \<.*UP.*\> ]]; then
      local interface_name="${BASH_REMATCH[1]}"
      # Debug output (can be removed in final version)
      # echo "  Interface found: $interface_name"
      local interface_up=true
      local is_loopback=false

      if [[ "$interface_name" == "lo" ]]; then
        is_loopback=true
        # Debug output (can be removed in final version)
        # echo "  Interface '$interface_name' is loopback. Skipping."
      fi

      if $is_loopback; then
        continue # Skip loopback interface
      fi

      # Look for inet lines within this interface block
      while IFS= read -r subnet_line; do
        # Debug output (can be removed in final version)
        # echo "  Processing subnet line: $subnet_line"
        if [[ "$subnet_line" =~ inet\ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/([0-9]+)\ .*scope\ ([^ ]+) ]]; then
          local ip_address="${BASH_REMATCH[1]}"
          local scope="${BASH_REMATCH[3]}"
          # Debug output (can be removed in final version)
          # echo "    Found inet line: IP Address: $ip_address, Scope: $scope"

          # Exclude loopback IPs and docker0/bridge related IPs by IP range
          if [[ "$ip_address" =~ ^127\. ]]; then
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is loopback. Skipping."
            continue # Skip 127.0.0.0/8 loopback IPs (although 'lo' should already be skipped)
          elif [[ "$ip_address" =~ ^169\.254\. ]]; then
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is link-local (169.254.x.x). Skipping."
            continue # Skip 169.254.0.0/16 link-local IPs (like docker0 often has)
          fi

          local is_private_ip=false
          if [[ "$ip_address" =~ ^10\.([0-9]{1,3}\.){2}[0-9]{1,3}$ ]] ||
             [[ "$ip_address" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]] ||
             [[ "$ip_address" =~ ^192\.168\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]]; then
            is_private_ip=true
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is a private IP."
          # else
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is NOT a private IP."
          fi

          if $is_private_ip || [[ "$scope" == "global" ]]; then # Consider private or global scope interfaces
            selected_interface="$interface_name"
            # Debug output (can be removed in final version)
            # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is selected."
            # echo "export GLOO_SOCKET_IFNAME=$selected_interface"
            # exit 0 # Exit immediately after finding the first suitable interface for debugging (removed for function)
            break 2 # Found a suitable interface! Break out of both inner and outer loops
          # else
            # Debug output (can be removed in final version)
            # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is NOT suitable (not private or global)."
          fi
        fi
      done < <(echo "$ip_output" | sed -n "/$interface_name: /,/^[0-9]\+:/p" | sed '$d' ) # Extract lines belonging to current interface block
      if [[ -n "$selected_interface" ]]; then # Check if selected_interface is not empty, if so, interface found and loops broken.
          # Debug output (can be removed in final version)
          # echo "      Selected interface '$selected_interface' already found. Breaking outer loop."
          break # Already found and assigned an interface, break outer loop as well.
      fi
    # else
      # Debug output (can be removed in final version)
      # echo "  Line does not match interface pattern."
    fi
  done < <(echo "$ip_output")

  if [[ -n "$selected_interface" ]]; then
    echo "$selected_interface"
  else
    echo "" # Return empty string if no interface is found, so export GLOO_SOCKET_IFNAME=  (empty)
    # echo "No suitable network interface could be automatically identified for GLOO_SOCKET_IFNAME." # No longer print error message to stderr in function context
    # return 1 # Optionally, you could return a non-zero exit code if you need to check for failure.
  fi
}

if [ -v VC_WORKER_HOSTS ]; then
    # Define a string
    
    # Set the IFS (Internal Field Separator) to space
    IFS=','

    # Use read to bash convert string to array
    read -ra myvar <<< "$VC_WORKER_HOSTS"

    # Output the array and its length
    
    echo "Number of elements in the vc worker hosts array: ${#myvar[@]}"
    WORLD_SIZE=${MA_NUM_HOSTS:-"1"}
    export RAY_MASTER_NODE_ADDRESS=${myvar[(($WORLD_SIZE-1))]}
    export RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-40000)
    

else 
    RAY_MASTER_NODE_ADDRESS="0.0.0.0"
    RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-65535)
    WORLD_SIZE=1
fi

cd $PROJECT_DIR
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

export NCCL_SOCKET_IFNAME=ens2f5
export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_DEBUG=INFO
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NODE_RANK="$VC_TASK_INDEX"
GPUS_PER_NODE="$MA_NUM_GPUS"
NNODES=$WORLD_SIZE
#export TORCH_USE_CUDA_DSA=1
#pip install pylatexenc vertexai sentence_transformers

export HF_ENDPOINT=https://hf-mirror.com

export SWANLAB_API_KEY='your_swanlab_token'

export MODEL_PATH=your_model_path/Meta-Llama-3.1-8B-Instruct


maxlen=${maxlen:-"4096"}
nsample=${nsample:-"16"}

PROJECT_NAME=sql_multi_turn
export EXPERIMENT_NAME=${1:-'test name'}

export VLLM_USE_V1=1
export PYTHONPATH="$PROJECT_DIR/:$PYTHONPATH"



if [ "$NODE_RANK" = "0" ]; then
    # Start Ray head node and capture the output
    rm -rf ip_tmp/${EXPERIMENT_NAME}
    mkdir -p ip_tmp/${EXPERIMENT_NAME}
    sleep 15
    ray_output=$(ray start --head --num-gpus 8)

    # Extract the IP address using grep and sed
    ip_address=$(echo "$ray_output" | grep -oP "ray start --address='\K[^']+")

    # Write the extracted IP address to a file named "ip.txt"

    echo "$ip_address" > ip_tmp/ip_${EXPERIMENT_NAME}.txt
    echo "$ip_address" > ip_tmp/ip.txt
    cat ip_tmp/ip_${EXPERIMENT_NAME}.txt

    # Example usage (to set the environment variable):
    export GLOO_SOCKET_IFNAME=$(find_interface)
    echo "$GLOO_SOCKET_IFNAME" > ip_tmp/gloo_${EXPERIMENT_NAME}.txt
        #    --eval_data /home/ma-user/work/haozhe/workspace/OpenRLHF/data/0209_eval_amcaime_queries \
    # if [ $nnode -gt 1 ]; then
    sleep 45
    
    # actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    # actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    # actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    # actor_rollout_ref.rollout.val_kwargs.n=8 \
    cd $PROJECT_DIR
    ray status
    python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='sql_single_grpo' \
        algorithm.adv_estimator=grpo \
       data.train_batch_size=128 \
       data.max_prompt_length=16000 \
        data.max_response_length=${maxlen} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.generate_sft=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.clip_ratio_high=0.2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=sync \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=${nsample} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        algorithm.use_kl_in_reward=False \
        trainer.val_before_train=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','swanlab'] \
        trainer.project_name='sql_async_rl' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=80 \
        trainer.test_freq=40 \
        trainer.default_local_dir=$PROJECT_DIR/rl_models/sql_llama8b/grpo_reversekl \
        data.train_files=$PROJECT_DIR//data/sql/llama3.1-8b/train_wrong.parquet \
        data.val_files=$PROJECT_DIR//data/sql/bird_dev.parquet \
        trainer.total_epochs=30

else 
    sleep 40 
    cd $PROJECT_DIR
       # Read the IP address from the file and assign it to the variable "head_ip"
    head_ip=$(cat ip_tmp/ip_${EXPERIMENT_NAME}.txt)
    gloo=$(cat ip_tmp/gloo_${EXPERIMENT_NAME}.txt)
    export GLOO_SOCKET_IFNAME=$gloo
    echo "gloo: $GLOO_SOCKET_IFNAME"
    # Print the value of head_ip for verification
    echo "Head IP Address: $head_ip"

    ray start --address ${head_ip}
    #echo $HOST_IP
    currect_ip=$(hostname -I | awk '{print $1}')
    echo "$currect_ip" > ip_tmp/${EXPERIMENT_NAME}/${NODE_RANK}.txt
    
     
    python rl/scorer/scorer_server_without_ray.py -c ./rl/scorer/sql.yaml
    #--address "$HOST_IP"
fi
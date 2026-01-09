#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=29550
NUM_NODES=1
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_document
# DATA_PATH=data/preprocessed_data/train

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 4 
    --hidden-size 4096 
    --num-attention-heads 32 
    --seq-length 2048 
    --max-position-embeddings 2048 
    # --attention-backend flash # Can use (flash/fused/unfused/local)
    # --transformer-impl local
    --no-masked-softmax-fusion # 禁用融合内核以避免编译问题
    --no-gradient-accumulation-fusion # 禁用梯度累积融合以避免编译问题
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 8 
    # --rampup-batch-size 16 16 5859375  # 注释掉：与 --train-iters 冲突
    --train-iters 50 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 4 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path data/preprocessed_data/train_content_document
    --split 949,50,1
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model tokenizer/Mixtral-8x7B
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 10
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

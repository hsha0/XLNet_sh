#!/bin/bash

#### local path
RACE_DIR=/content/RACE
INIT_CKPT_DIR='xlnet_cased_L-12_H-768_A-12'

#### google storage path
GS_ROOT=gs://xlnet_sh
GS_INIT_CKPT_DIR=${GS_ROOT}/${INIT_CKPT_DIR}
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/race
GS_MODEL_DIR=${GS_ROOT}/experiment/race_epochs2_lr2e-05

# TPU name in google cloud
TPU_NAME=grpc://10.111.207.226:8470

for run in $(seq 7 10)
do
    python3 run_race.py \
      --use_tpu=True \
      --tpu=${TPU_NAME} \
      --num_hosts=1 \
      --num_core_per_host=8 \
      --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
      --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
      --output_dir=${GS_PROC_DATA_DIR} \
      --init_checkpoint=${GS_INIT_CKPT_DIR}/xlnet_model.ckpt \
      --model_dir=${GS_MODEL_DIR}_$run \
      --data_dir=${RACE_DIR} \
      --max_seq_length=512 \
      --max_qa_length=128 \
      --uncased=False \
      --do_train=True \
      --train_batch_size=8 \
      --do_eval=True \
      --eval_batch_size=32 \
      --train_steps=12000 \
      --save_steps=1000 \
      --iterations=1000 \
      --warmup_steps=1000 \
      --learning_rate=2e-5 \
      --weight_decay=0 \
      --adam_epsilon=1e-6 \
      --middle_only=True \
      --num_train_epochs=2 \
      $@
done

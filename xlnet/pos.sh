#!/bin/bash

#### local path
POS_DIR=/content/POS
INIT_CKPT_DIR='xlnet_cased_L-24_H-1024_A-16'

#### google storage path
GS_ROOT=gs://xlnet_sh
GS_INIT_CKPT_DIR=${GS_ROOT}/${INIT_CKPT_DIR}
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/pos
GS_MODEL_DIR=${GS_ROOT}/experiment/pos/pos

# TPU name in google cloud
TPU_NAME=grpc://10.14.32.90:8470

for run in $(seq 0 9)
do
    python3 run_pos.py \
      --use_tpu=True \
      --tpu=${TPU_NAME} \
      --num_hosts=1 \
      --num_core_per_host=8 \
      --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
      --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
      --output_dir=${GS_PROC_DATA_DIR} \
      --init_checkpoint=${GS_INIT_CKPT_DIR}/xlnet_model.ckpt \
      --model_dir=${GS_MODEL_DIR}_$run \
      --data_dir=${POS_DIR} \
      --train_file=tagged.large.refined \
      --test_file=heldback \
      --max_seq_length=512 \
      --uncased=False \
      --do_train=True \
      --train_batch_size=32 \
      --do_eval=True \
      --eval_batch_size=32 \
      --save_steps=1000 \
      --iterations=1000 \
      --warmup_steps=1000 \
      --learning_rate=5e-5 \
      --weight_decay=0 \
      --adam_epsilon=1e-6 \
      --middle_only=True \
      --num_train_epochs=3 \
      $@
done
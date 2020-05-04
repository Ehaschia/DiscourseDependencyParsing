#!/usr/bin/env sh

GPU=0
MODEL=arcfactoredmodel
CONFIG=./config/hyperparams_1.ini
NAME=trial1

# Training with the full training set
MAX_EPOCH=15
python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype train \
    --max_epoch ${MAX_EPOCH}

# Evaluation
python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype evaluate

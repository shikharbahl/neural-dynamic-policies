#!/bin/bash


ENV_NAME=$1
TYPE=$2
EXP_ID=$3
SEED=$4
num_steps=150
num_processes=5

cleanup() {
    exit 1
}

if [ "$ENV_NAME" = "throw" ] || [ "$ENV_NAME" = "pick" ]; then
    num_steps=55
fi

python ./vis_ndp_policy.py --env-name $ENV_NAME  --type $TYPE --seed $SEED --expID $EXP_ID

wait $!

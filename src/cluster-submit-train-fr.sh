#!/bin/bash

exp_name=free-recall
penalty=-.5
n=6
n_std=20
lr=1e-3
n_epochs=200001
reward=1
penalize_repeat=1

for subj_id in {0..3}
do
   for dim_hidden in 64 512 2048
   do
         sbatch train-model.sh \
         ${exp_name} ${subj_id} ${n} ${n_std} ${dim_hidden} \
         ${lr} ${n_epochs} ${reward} ${penalty} ${penalize_repeat}
   done
done

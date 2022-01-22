#!/bin/bash

exp_name=free-recall
reward=1
penalty=-.5
penalize_repeat=1

lr=1e-3
n_epochs=200001

n_std=6

for subj_id in {0..3}
do
   for dim_hidden in 64 512 2048
   do
     for n in 20
     do
         sbatch cluster-train-fr.sh \
         ${exp_name} ${subj_id} ${n} ${n_std} ${dim_hidden} \
         ${lr} ${n_epochs} ${reward} ${penalty} ${penalize_repeat}
       done
   done
done

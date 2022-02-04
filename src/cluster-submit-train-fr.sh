#!/bin/bash

exp_name=free-recall
penalize_repeat=1

lr=1e-3
n_epochs=200001

n_std=8

for subj_id in {0..5}
do
   for dim_hidden in 128 512 1024
   do
     for n in 20
     do
         sbatch cluster-train-fr.sh \
         ${exp_name} ${subj_id} ${n} ${n_std} ${dim_hidden} \
         ${lr} ${n_epochs} ${penalize_repeat}
       done
   done
done

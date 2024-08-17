#!/bin/zsh

python3 src/train_ampliy_union.py \
    -amp_act_tr ./data_union/train_activity_AMP.fa \
    -non_amp_act_tr ./data_union/train_activity_nonAMP.fa \
    -amp_tox_tr ./data_union/train_toxicity_AMP.fa \
    -non_amp_tox_tr ./data_union/train_toxicity_nonAMP.fa \
    -out_dir ./experiment_2 \
    -model_name amplify_union


# amp_act_tr
# non_amp_act_tr
# -amp_act_te
# -non_act_amp_te
# amp_tox_tr
# non_amp_tox_tr
# - amp_tox_te
# - non_tox_amp_te
# out_dir
# model_name
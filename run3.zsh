#!/bin/zsh
python3 src/train_amplify_duo.py \
    -amp_act_tr ./data_test/train_activity_AMP.fa \
    -non_amp_act_tr ./data_test/train_activity_nonAMP.fa \
    -amp_tox_tr ./data_test/train_toxicity_AMP.fa \
    -non_amp_tox_tr ./data_test/train_toxicity_nonAMP.fa \
    -out_dir ./experiment_3 \
    -model_name amplify_duo

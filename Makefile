help: ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

venv: ## Create/Update venv
	uv venv
	. .venv/bin/activate
	uv pip compile pyproject.toml -o requirements.txt
	uv pip install -r requirements.txt
	
duo:
	python src/train_amplify_duo.py \
        -amp_act_tr ./data_test/train_activity_AMP.fa \
        -non_amp_act_tr ./data_test/train_activity_nonAMP.fa \
        -amp_tox_tr ./data_test/train_toxicity_AMP.fa \
        -non_amp_tox_tr ./data_test/train_toxicity_nonAMP.fa \
        -out_dir ./experiment_3 \
        -model_name amplify_duo

union:
	python src/train_ampliy_union.py \
        -amp_act_tr ./data_union/train_activity_AMP.fa \
        -non_amp_act_tr ./data_union/train_activity_nonAMP.fa \
        -amp_tox_tr ./data_union/train_toxicity_AMP.fa \
        -non_amp_tox_tr ./data_union/train_toxicity_nonAMP.fa \
        -out_dir ./experiment_2 \
        -model_name amplify_union
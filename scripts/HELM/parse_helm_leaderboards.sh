uv run python3 -m automation_scripts.helm.adapter --leaderboard_name HELM_Capabilities --source_data_url https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.12.0/groups/core_scenarios.json

uv run python3 -m automation_scripts.helm.adapter --leaderboard_name HELM_Lite --source_data_url https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/groups/core_scenarios.json

uv run python3 -m automation_scripts.helm.adapter --leaderboard_name HELM_Classic --source_data_url https://storage.googleapis.com/crfm-helm-public/benchmark_output/releases/v0.4.0/groups/core_scenarios.json 

uv run python3 -m automation_scripts.helm.adapter --leaderboard_name HELM_Instruct --source_data_url https://storage.googleapis.com/crfm-helm-public/instruct/benchmark_output/releases/v1.0.0/groups/instruction_following.json

uv run python3 -m automation_scripts.helm.adapter --leaderboard_name HELM_MMLU --source_data_url https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/releases/v1.13.0/groups/mmlu_subjects.json
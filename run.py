import os
import time
import glob
from src.utils import load_json, save_json

total_jobs = 0
gpu_count = 0
def run(model_name_or_path, result_path, config_path, method, data_num=100, name="test"):
    global gpu_count, total_jobs, gpus
    gpu_id = gpus[gpu_count]
    gpu_count = (gpu_count + 1) % len(gpus)
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={str(gpu_id)}"
    model_name = model_name_or_path.replace('/', '-')
    log_file = f"./logs/main_logs/{name}/{model_name}-{result_path.split('/')[-1].split('.')[0]}-{method}-{config_path.split('/')[-1].split('.')[0]}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    cmd = f"({gpu_cmd} && python3 -u main.py \
            --model_name_or_path {model_name_or_path} \
            --result_path {result_path} \
            --name {name} \
            --config_path {config_path} \
            --method {method} \
            --data_num {data_num} \
            > {log_file} 2>&1) &"
    os.system(cmd)
    print(cmd)
    total_jobs += 1
    return 1


name = "pisanitizer"
gpus = [0, 1, 2, 3]

# result_paths = glob.glob(f"data/LongBench_injection/*.json")
result_paths = [
    "data/LongBench_injection/incorrect-qasper-combine-random-1.json",
    "data/LongBench_injection/incorrect-hotpotqa-combine-random-1.json",
    "data/LongBench_injection/incorrect-gov_report-combine-random-1.json",
    "data/LongBench_injection/incorrect-multi_news-combine-random-1.json",
    "data/LongBench_injection/incorrect-lcc-combine-random-1.json",
    "data/LongBench_injection/incorrect-passage_retrieval_en-combine-random-1.json",
]


method = "pisanitizer"
config_paths = []
config = {
    "mode": "max-avg",
    "start_offset": 500,
    "end_offset": 500,
    "anchor_prompt": 0,
    "smooth_win": None,
    "max_gap": 10,
    "threshold": 0.01
}
config_name = f"max-avg-{config['smooth_win']}-{config['max_gap']}-{config['threshold']}"
config_path = f"configs/method_configs/{name}/{method}/{config_name.replace('.', '_')}.json"
save_json(config, config_path)
config_paths.append(config_path)

for model in ["meta-llama/Llama-3.1-8B-Instruct"]:
    model_name = model.replace("/", "-")
    for config_path in config_paths:
        for result_path in result_paths:
            run(
                model,
                result_path,
                config_path=config_path,
                method=config_path.split("/")[-2],
                data_num=100,
                name=name,
            )

print(f"Started {total_jobs} jobs in total")
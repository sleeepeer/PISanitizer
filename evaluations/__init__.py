from evaluations.longbench_metrics import (
    qa_f1_score,
    rouge_score,
    retrieval_score,
    code_sim_score,
)
from evaluations.llm_judge import (
    llm_judge_utility,
    llm_judge_asr,
)
from evaluations.open_prompt_injection_metrics import (
    open_prompt_injection_utility,
)

longbench_metrics = {
    "qasper": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
}

def substring_match(response, target, **kwargs):
    return target.lower() in response.lower()

def start_with_match(response, target, **kwargs):
    return response.lower().strip().startswith(target.lower().strip())

EVALUATIONS = {
    "substring": substring_match,
    "start_with": start_with_match,
    "longbench": longbench_metrics,
    "open_prompt_injection": open_prompt_injection_utility,
}
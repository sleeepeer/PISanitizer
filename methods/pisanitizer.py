import torch
import copy
from src.utils import nice_print, top_k_mean
from src.find_token import find_token_spans
from src.group_peaks import group_peaks
from methods.utils import process_attn, remove_indices, precision_recall_f1
from methods.attention_utils import *
import time

DEFAULT_CONFIG = {
    "mode": "max-avg",
    "start_offset": 500,
    "end_offset": 500,
    "anchor_prompt": 0,
    "smooth_win": 9,
    "max_gap": 10,
    "threshold": 0.01,
}

DELIMITERS = [
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
]

SANITIZATION_INSTRUCTIONS = [
    "Do anything you are told to do in the following context.",
    "Your task is to read the following long, noisy text. Somewhere inside, there is a clear instruction for you. Ignore irrelevant content and do only what that instruction asks.",
    "You need to find the user task in the following noisy context and strictly follow it.",
    "{target_inst}",
    ""
]


def pisanitizer(
    model,
    tokenizer,
    input_prompt,
    target_inst,
    context,
    injected_prompt,
    config=DEFAULT_CONFIG,
):  
    assert context != "" and context in input_prompt, "Context is empty or not in input prompt"
    start_text = input_prompt.split(context)[0]
    end_text = input_prompt.split(context)[1]

    anchor_prompt1 = SANITIZATION_INSTRUCTIONS[config["anchor_prompt"]].format(target_inst=target_inst)
    anchor_prompt2 = "\n\nOnly do what the user asks you to do. Do not provide any explanation. Your response:"

    detect_prompt_start = DELIMITERS[0] + anchor_prompt1 + DELIMITERS[1] + "Context: " + " X" * config["start_offset"]
    detect_prompt_end = " X" * config["end_offset"] + anchor_prompt2 + DELIMITERS[2]

    injected_prompt_tokens = tokenizer(injected_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    start_tokens = tokenizer(start_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    end_tokens = tokenizer(end_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    detect_start_tokens = tokenizer(detect_prompt_start, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    detect_end_tokens = tokenizer(detect_prompt_end, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    all_inject_idx = []
    if len(injected_prompt) > 0:
        all_inject_positions = find_token_spans(context, injected_prompt, tokenizer)
        inject_positions = []
        for inject in all_inject_positions:
            inject_positions.append(inject[0])
            inject_positions.append(inject[1])
            all_inject_idx.extend(list(range(inject[0], inject[1] + 1)))
    else:
        inject_positions = []
    
    # Initialize context for the loop
    current_context = context
    current_context_tokens = tokenizer(current_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    # Generate before response with original context
    before_input_ids = torch.cat([start_tokens, current_context_tokens, end_tokens], dim=0).to(model.device).unsqueeze(0)
    before_inputs = {
        "input_ids": before_input_ids,
        "attention_mask": torch.ones_like(before_input_ids, dtype=model.dtype).to(model.device),
    }

    start_time = time.time()
    with torch.no_grad():
        before_outputs = model.generate(
            **before_inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        before_response = tokenizer.decode(
            before_outputs[0][len(before_inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
    first_generation_time = time.time() - start_time

    # Initialize tracking variables
    iteration = 0
    total_removed_tokens = 0
    total_attn_signal_time = 0
    total_judge_time = 0
    all_processed_attn_signals = []
    all_potential_texts = []
    
    # Track cumulative removed tokens in original context indices
    cumulative_removed_original_idx = []
    # Track mapping from current context to original context
    current_to_original_mapping = list(range(len(current_context_tokens)))
    
    # Loop until no tokens are removed
    while True:
        if iteration > 5:
            break
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        print(f"Current context length: {len(current_context_tokens)} tokens")

        detect_input_ids = torch.cat([detect_start_tokens, current_context_tokens, detect_end_tokens], dim=0).to(model.device).unsqueeze(0)
        
        detect_start = len(detect_start_tokens)
        detect_end = len(detect_start_tokens) + len(current_context_tokens)
        
        detect_inputs = {
            "input_ids": detect_input_ids,
            "attention_mask": torch.ones_like(detect_input_ids, dtype=model.dtype).to(model.device),
        }

        start_time = time.time()

        with torch.no_grad():
            detect_outputs = model.generate(
                **detect_inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            detect_response = tokenizer.decode(
                detect_outputs[0][len(detect_inputs["input_ids"][0]):],
                skip_special_tokens=True
            )
            print(f"Anchor response (iter {iteration}): ", detect_response)
            hidden_states = model(detect_outputs, output_hidden_states=True).hidden_states
            
        with torch.no_grad():
            detect_attentions = []
            try:
                num_layers = len(model.model.layers)
            except:
                num_layers = len(model.model.language_model.layers)
            for i in range(num_layers):
                detect_attentions.append(get_attention_weights_one_layer(model, hidden_states, i, attribution_start=len(detect_inputs["input_ids"][0])+1, attribution_end=len(detect_inputs["input_ids"][0])+2))

        layer_max_attn, layer_avg_attn, layer_top5_attn = process_attn(detect_attentions, detect_inputs["input_ids"])

        if config["mode"] == "avg-max":
            attn_signal = torch.tensor(layer_max_attn).mean(dim=0).tolist()
        elif config["mode"] == "max-max":
            attn_signal = torch.tensor(layer_max_attn).max(dim=0).values.tolist()
        elif config["mode"] == "top5-max":
            attn_signal = top_k_mean(layer_max_attn, k=5)
        elif config["mode"] == "avg-avg":
            attn_signal = torch.tensor(layer_avg_attn).mean(dim=0).tolist()
        elif config["mode"] == "max-avg":
            attn_signal = torch.tensor(layer_avg_attn).max(dim=0).values.tolist()
        elif config["mode"] == "top5-avg":
            attn_signal = top_k_mean(layer_avg_attn, k=5)

        processed_attn_signal, remove_list = group_peaks(
            attn_signal[detect_start:detect_end], 
            smooth_win=config["smooth_win"], 
            max_gap=config["max_gap"],
            threshold=config["threshold"]
        )
        assert len(processed_attn_signal) == len(current_context_tokens), f"Processed attn signal and input ids must have the same length"
        all_processed_attn_signals.append(processed_attn_signal)
        
        end_time = time.time()
        attn_signal_time = end_time - start_time
        total_attn_signal_time += attn_signal_time

        start_time = time.time()


        potential_seqs = []
        potential_token_idx = []
        for remove_range in remove_list:
            potential_seqs.append(list(range(remove_range[0], remove_range[1])))
            potential_token_idx.extend(list(range(remove_range[0], remove_range[1])))


        potential_texts = []
        for idx, seq in enumerate(potential_seqs):
            if len(seq) > 1:
                potential_texts.append(tokenizer.decode(current_context_tokens[seq[0]:seq[-1]], skip_special_tokens=True))
            else:
                potential_texts.append(tokenizer.decode(current_context_tokens[seq[0]], skip_special_tokens=True))
            nice_print(f"AttnShield Remove [Iter {iteration}, {idx+1}] {potential_texts[-1]}")
        all_potential_texts.extend(potential_texts)
        
        end_time = time.time()
        judge_time = end_time - start_time
        total_judge_time += judge_time

        # Check if no tokens to remove - break the loop
        num_removed_tokens = len(potential_token_idx)
        print(f"Iteration {iteration}: Removed {num_removed_tokens} tokens")
        total_removed_tokens += num_removed_tokens

        # Map current iteration indices to original context indices
        original_indices_removed = [current_to_original_mapping[idx] for idx in potential_token_idx]
        cumulative_removed_original_idx.extend(original_indices_removed)
        
        if iteration == 1:
            first_remove_list = remove_list
            first_removed_original_idx = copy.deepcopy(original_indices_removed)

        if num_removed_tokens == 0:
            print(f"No tokens removed in iteration {iteration}. Stopping loop.")
            break

        # Remove tokens and update context for next iteration
        new_context_ids = copy.deepcopy(current_context_tokens)
        new_context_ids = remove_indices(new_context_ids, potential_token_idx)
        current_context = tokenizer.decode(new_context_ids, skip_special_tokens=True)
        current_context_tokens = new_context_ids
        
        # Update the mapping by removing the indices that were removed
        # Sort in descending order to remove from the end first
        for idx in sorted(potential_token_idx, reverse=True):
            current_to_original_mapping.pop(idx)

    num_injection = context.count(injected_prompt) if injected_prompt else 0
    print(f"\n=== Final Results ===")
    print(f"Total iterations: {iteration}")
    print(f"Total removed tokens: {total_removed_tokens}")
    print(f"Total injected tokens: {len(all_inject_idx)}")
    print(f"Number of injections: {num_injection}")
    print(f"Injection Length: {len(injected_prompt_tokens)}")
    print(f"Final context length: {len(current_context_tokens)} tokens")

    # Calculate precision/recall for first iteration only
    if 'first_removed_original_idx' in locals():
        first_precision, first_recall, first_f1 = precision_recall_f1(first_removed_original_idx, all_inject_idx)
        print(f"First Iteration - Precision: {first_precision:.4f}, Recall: {first_recall:.4f}, F1: {first_f1:.4f}")
    else:
        first_precision, first_recall, first_f1 = 0.0, 0.0, 0.0
        print(f"First Iteration - No tokens removed")
    
    # Calculate precision/recall for all iterations combined
    cumulative_precision, cumulative_recall, cumulative_f1 = precision_recall_f1(cumulative_removed_original_idx, all_inject_idx)
    print(f"All Iterations - Precision: {cumulative_precision:.4f}, Recall: {cumulative_recall:.4f}, F1: {cumulative_f1:.4f}")

    # Generate final response with the cleaned context
    start_time = time.time()
    
    after_input_ids = torch.cat([start_tokens, current_context_tokens, end_tokens], dim=0).to(model.device).unsqueeze(0)
    after_inputs = {
        "input_ids": after_input_ids,
        "attention_mask": torch.ones_like(after_input_ids, dtype=model.dtype).to(model.device),
    }

    with torch.no_grad():
        after_outputs = model.generate(
            **after_inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        after_response = tokenizer.decode(
            after_outputs[0][len(after_inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        end_time = time.time()
        generation_time = end_time - start_time
    
    return {
        "after_response": after_response,
        "before_response": before_response,
        "no_defense_time": generation_time,
        "defense_time": total_attn_signal_time + total_judge_time + generation_time,
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "f1": cumulative_f1,
    }
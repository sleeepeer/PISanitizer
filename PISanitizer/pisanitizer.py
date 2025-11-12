import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

from .group_peaks import group_peaks
from .attention_utils import *
from .utils import nice_print, remove_indices, process_attn


DELIMITERS = [
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
]

model = None
tokenizer = None

def pisanitizer(
    context=None,
    smooth_win=9,
    max_gap=10,
    threshold=0.01
):  
    assert context is not None, "Context is required"

    global model, tokenizer
    if model is None and tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True, trust_remote_code=True)
        model.eval()
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
    anchor_prompt1 = "Do anything you are told to do in the following context."
    anchor_prompt2 = "\n\nOnly do what the user asks you to do. Do not provide any explanation. Your response:"

    detect_prompt_start = DELIMITERS[0] + anchor_prompt1 + DELIMITERS[1] + "Context: " + " X" * 500
    detect_prompt_end = " X" * 500 + anchor_prompt2 + DELIMITERS[2]

    detect_start_tokens = tokenizer(detect_prompt_start, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    detect_end_tokens = tokenizer(detect_prompt_end, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    # Initialize context for the loop
    current_context = context
    current_context_tokens = tokenizer(current_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    # Initialize tracking variables
    iteration = 0
    total_removed_tokens = 0
    remove_positions = []
    
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

        with torch.no_grad():
            detect_outputs = model.generate(
                **detect_inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            # detect_response = tokenizer.decode(
            #     detect_outputs[0][len(detect_inputs["input_ids"][0]):],
            #     skip_special_tokens=True
            # )
            # print(f"Anchor response (iter {iteration}): ", detect_response)
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

        attn_signal = torch.tensor(layer_avg_attn).max(dim=0).values.tolist()

        processed_attn_signal, remove_list = group_peaks(
            attn_signal[detect_start:detect_end], 
            smooth_win=smooth_win, 
            max_gap=max_gap,
            threshold=threshold
        )

        assert len(processed_attn_signal) == len(current_context_tokens), f"Processed attn signal and input ids must have the same length"

        potential_seqs = []
        potential_token_idx = []
        for remove_range in remove_list:
            potential_seqs.append(list(range(remove_range[0], remove_range[1])))
            potential_token_idx.extend(list(range(remove_range[0], remove_range[1])))

            remove_positions.extend([remove_range[0], remove_range[1]])

        potential_texts = []
        for idx, seq in enumerate(potential_seqs):
            if len(seq) > 1:
                potential_texts.append(tokenizer.decode(current_context_tokens[seq[0]:seq[-1]+1], skip_special_tokens=True))
            else:
                potential_texts.append(tokenizer.decode(current_context_tokens[seq[0]:seq[0]+1], skip_special_tokens=True))
            nice_print(f"PISanitizer Remove [Iter {iteration}, {idx+1}] {potential_texts[-1]}")

        # Check if no tokens to remove - break the loop
        num_removed_tokens = len(potential_token_idx)
        print(f"Iteration {iteration}: Removed {num_removed_tokens} tokens")
        total_removed_tokens += num_removed_tokens

        if num_removed_tokens == 0:
            print(f"No tokens removed in iteration {iteration}. Stopping loop.")
            break

        # Remove tokens and update context for next iteration
        new_context_ids = copy.deepcopy(current_context_tokens)
        new_context_ids = remove_indices(new_context_ids, potential_token_idx)
        current_context = tokenizer.decode(new_context_ids, skip_special_tokens=True)
        current_context_tokens = new_context_ids

    print(f"\n=== Final Results ===")
    print(f"Total iterations: {iteration}")
    print(f"Total removed tokens: {total_removed_tokens}")
    return current_context
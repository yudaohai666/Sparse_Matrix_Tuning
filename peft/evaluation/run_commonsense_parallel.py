'''
Example command:
accelerate launch \
  --main_process_port 64941 \
  --gpu_ids '0,1,2,3' \
  evaluation/run_commonsense_parallel.py \
  --data_path /home/sidaw/Projects/llm/LLM-FT/data/commonsense/dataset/ \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --tokenizer_path meta-llama/Meta-Llama-3-8B \
  --per_device_eval_batch_size 4 \
  --seed 1234 \
  --dtype bf16 \
  --dataset boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande \
  --output_dir tmp
'''

# Source code modified from 
# 1. https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/multi_dataset_eval.py;
# 2.https://github.com/aksh555/LoRA-Soups/blob/main/evaluate.py; 
# 3. https://github.com/aksh555/LoRA-Soups/blob/main/utils.py
# 4. https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/commonsense_evaluate.py;

import argparse
import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import re
import json
import torch
from helpers.deepspeed_helpers import (
    print_rank_0,
    synchronize_index_list,
    to_device,
    save_hf_format,
    set_random_seed,
    create_hf_model,
    get_optimizer_grouped_parameters,
    load_hf_tokenizer,
    print_throughput,
    make_model_gradient_checkpointing_compatible,
)

from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
)
import argparse
import deepspeed
from accelerate import Accelerator
from accelerate.utils import gather_object


i_prompt = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
'''


def extract_answer(dataset, sentence: str) -> float:
    sentence = sentence.lower()
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5',
                                  sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4',
                                  sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

@torch.no_grad()
def generate_text_completions(
        model, device, tokenizer, prompt_list,
        batch_size=1, stop_sequences=None, disable_progress=False,
        verbose=False, **generation_options
):
    """
    Generate text completions for a list of prompts using a pre-trained model.

    Args:
        model: The pre-trained model for text generation.
        device: The device (e.g., 'cpu' or 'cuda') to execute the model on.
        tokenizer: The tokenizer associated with the model.
        prompt_list: A list of prompts for which completions are needed.
        batch_size: Number of prompts to process in one batch.
        stop_sequences: List of sequences that signal when to stop generation.
        disable_progress: If True, disables the progress bar display.
        verbose: If True, outputs intermediate results for debugging.
        generation_options: Additional options for text generation.

    Returns:
        A list of generated text completions, one per prompt in the input list.
    """
    results = []

    # Initialize progress bar if enabled
    progress_bar = None
    if not disable_progress:
        progress_bar = tqdm.tqdm(total=len(prompt_list), desc="Generating Text")

    # Retrieve number of return sequences per prompt
    return_sequences_count = generation_options.get("num_return_sequences", 1)

    # Process prompts in batches
    for start_idx in range(0, len(prompt_list), batch_size):
        # Extract current batch of prompts
        current_batch = prompt_list[start_idx:start_idx + batch_size]

        # Tokenize the batch of prompts
        token_data = tokenizer(current_batch, padding='longest', return_tensors="pt")
        input_ids = token_data.input_ids.to(device)
        attention_masks = token_data.attention_mask.to(device)

        try:
            # Generate text completions for the current batch
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_sequences)] if stop_sequences else None,
                **generation_options
            )
            generated_outputs = generated_outputs.detach().cpu()

            # Apply stopping criteria to trim unwanted tokens
            if stop_sequences:
                for seq_idx in range(generated_outputs.shape[0]):
                    for token_idx in range(input_ids.shape[1], generated_outputs.shape[1]):
                        if any(generated_outputs[seq_idx, token_idx: token_idx + len(seq)].tolist() == seq for seq in
                               stop_sequences):
                            generated_outputs[seq_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # Replace invalid tokens with unknown token ID
            generated_outputs[generated_outputs >= tokenizer.vocab_size] = tokenizer.unk_token_id
            generated_outputs[generated_outputs == -1] = tokenizer.unk_token_id

            # Decode both the outputs and prompts to text
            decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            decoded_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # Duplicate prompts to match the number of return sequences per prompt
            repeated_prompts = [prompt for prompt in decoded_prompts for _ in range(return_sequences_count)]

            # Extract the generated content by removing the prompt prefix
            batch_results = [
                text[len(prompt):] for prompt, text in zip(repeated_prompts, decoded_outputs)
            ]
        except Exception as error:
            # Handle any errors in generation by returning empty results for the batch
            batch_results = [""] * len(current_batch) * return_sequences_count

        # Append the results of the current batch to the final list
        results.extend(batch_results)

        # Update the progress bar if enabled
        if progress_bar:
            progress_bar.update(len(current_batch) // return_sequences_count)

    # Close progress bar if it was used
    if progress_bar:
        progress_bar.close()

    # Ensure the number of results matches expectations
    assert len(results) == len(prompt_list) * return_sequences_count, (
        "Mismatch in the number of results and expected completions"
    )
    return results


@torch.no_grad()
def main(args):
    accelerator = Accelerator()
    set_random_seed(args.seed)

    print_rank_0("Loading model and tokenizer...")
    tokenizer = load_hf_tokenizer(args.tokenizer_path, fast_tokenizer=True)

    if '8b' in args.model_name_or_path.lower():
        tokenizer.unk_token_id =0
        # tokenizer.padding_side = "right"

    tokenizer.padding_side = "left"
    print_rank_0(f"tokenizer pad side: {tokenizer.padding_side}")

    ### create_hf_trained_model, use model local path
    # model = create_hf_trained_model(AutoModelForCausalLM,
    #                     args.model_name_or_path,
    #                     tokenizer,
    #                     ds_config=None,
    #                     dropout=args.dropout)

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            dropout=args.dropout)
    model = model.to(accelerator.device)
    args.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    model = model.to(args.dtype)
    model.eval()
    print_rank_0('model is dtype: {}'.format(model.dtype))


    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=4,  # Thorough search
        temperature=0.0,  # No randomness
        repetition_penalty=1.1,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )


    for dataset in args.datasets:
        print_rank_0(f"Handling dataset: {dataset}")
        t_test_data = json.load(open(os.path.join(args.data_path, dataset, 'test.json'), 'r'))

        prompts = []
        for example in t_test_data:
            prompt = i_prompt.format_map(example)
            prompts.append(prompt)
        print_rank_0(prompts[0])

        accelerator.wait_for_everyone()
        device = accelerator.device
        with accelerator.split_between_processes(prompts) as prompt:
            model_outputs = []
            outputs = generate_text_completions(
                model=model,
                device=device,
                tokenizer=tokenizer,
                prompts=prompt,
                max_new_tokens=256,
                batch_size=args.per_device_eval_batch_size,
                stop_id_sequences=[[tokenizer.eos_token]],
                verbose=False,
                generation_config=generation_config)
            model_outputs.extend(outputs)
        outputs = gather_object(model_outputs)

        save_outputs = []
        correct = 0
        for example, output in zip(t_test_data, outputs):
            example['raw_output'] = output
            target = example["answer"].lower()
            predict = extract_answer(dataset, output)
            if target == predict:
                correct += 1
            example['prediction'] = predict
            save_outputs.append(example)

        print_rank_0(f"Saving outputs to {args.output_dir}")

        weighted_acc = correct / len(t_test_data)
        print_rank_0("Dataset: {}, accuracy {:.1f}%, number of test data: {}".format(
            dataset, 
            weighted_acc * 100,
            len(t_test_data)
        ))
        
        dataset_output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True) 
        with open(os.path.join(dataset_output_dir, f"model_predictions.jsonl"),
                "w") as fout:
            for example in save_outputs:
                fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        default="my_data/mmlu",
                        required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",  # Accepts 1 or more values
        default=["boolq"],
        help="List of dataset names"
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default="eval/gsm",
                        required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to tokenizer identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='dropout rate of the model.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Inference data type')
    parser.add_argument("--per_device_eval_batch_size",
                        type=int,
                        default=4,
                        help="batch size for evaluation.")
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help=
        "If given, the prompt will be encoded as a chat format with the roles in prompt."
    )
    args = parser.parse_args()

    #args.output_dir = os.path.join(args.output_dir, '-'.join(args.model.split('/')[-2:]))

    main(args)

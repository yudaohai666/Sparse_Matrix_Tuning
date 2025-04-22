
# LLM SMT Fine-Tuning with Transformer-PEFT

## **Environment Setup**

### Prerequisites
- CUDA 11.8
- GPU: H100, A100, or A6000 (BF16/FP16 support recommended)
- Conda package manager

### Installation

1. Create conda environment from YAML:
   ```bash
   conda env create -f peft_environment.yml -n peft_env
   conda activate peft_env
   ```

2. Manual package fixes (if needed)
   Due to environment variability, some dependencies may fail to install cleanly from deepspeed_environment.yml. Here's how to handle that:
   - Check CUDA compatibility: Verify that the installed PyTorch version matches your CUDA version.
   - Manually install missing packages (possibly Deepspeed, Transformers, etc) using pip install `<package-name>`.
   - Example command that partially fixes the packages:
     ```bash
     deepspeed==0.13.1
     ```

## Datasets

Please download the dataset: [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/ft-training_set).


# Example Commands
## Evaluation using deepspeed/accelerate environment

```
accelerate launch \
  --main_process_port 64941 \
  --gpu_ids '0,1,2,3' \
  evaluation/run_commonsense_parallel.py \
  --data_path ../data/commen_sense/dataset/ \
  --model_name_or_path /ocean/projects/cis250057p/hhe4/LLM-FT/deepspeed/logs/DeepSeek-R1-Distill-Llama-8B_04020_0736_smt/epoch_1 \
  --tokenizer_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --per_device_eval_batch_size 4 \
  --seed 1234 \
  --dtype bf16 \
  --dataset boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande \
  --output_dir logs/eval/DeepSeek-R1-Distill-Llama-8B_04020_0736_smt \
  > logs/eval/DeepSeek-R1-Distill-Llama-8B_04020_0736_smt.log 2>&1
```


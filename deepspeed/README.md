# LLM SMT Fine-Tuning with DeepSpeed

## **Environment Setup**

### Prerequisites
- CUDA 11.8
- GPU: H100, A100, or A6000 (BF16/FP16 support recommended)
- Conda package manager

### Installation

1. Create conda environment from YAML:
   ```bash
   conda env create -f deepspeed_environment.yml -n deepspeed_env
   conda activate deepspeed_env
   ```

2. Manual package fixes (if needed)
   Due to environment variability, some dependencies may fail to install cleanly from deepspeed_environment.yml. Here's how to handle that:
   - Check CUDA compatibility: Verify that the installed PyTorch version matches your CUDA version.
   - Manually install missing packages (possibly Deepspeed, Transformers, etc) using pip install `<package-name>`.
   - Example command that partially fixes the packages:
     ```bash
     pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 transformers==4.38.1 deepspeed==0.13.1 pytorch_memlab starlette==0.45.0 uvicorn==0.27.0 --index-url https://download.pytorch.org/whl/cu118
     ```

## Datasets

Please download the dataset: [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/ft-training_set).


## Example Commands

### Training
```
deepspeed --include=localhost:0,1,2,3 --master_port 64940 fine_tune.py \
    --offload \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --per_device_ft_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_seq_len 2048 \
    --ft_learning_rate 9.865e-6 \
    --num_ft_epochs 3 \
    --lr_warmup_steps 100 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --smt_lr 9.865e-6 \
    --eval_step 30 \
    --eval_set_ratio 0.2 \
    --matrix_sparsity \
    --selection_strategy "no_restriction" \
    --calculate_strategy "abs_mean" \
    --downsample_mlp_blocks_ratio 0.0084 \
    --downsample_attention_blocks_ratio 0.0084 \
    --num_mlp_channel 0 \
    --num_attention_channel 0 \
    --full_ft_steps 100 \
    --smt_lr_warmup_steps 0 \
    --data_path ../data/commen_sense/ft-training_set/commonsense_170k.json \
    --output_dir "/ocean/projects/cis250057p/hhe4/LLM-FT/deepspeed/logs/DeepSeek-R1-Distill-Llama-8B_04020_2034_smt/" \
    > "/ocean/projects/cis250057p/hhe4/LLM-FT/deepspeed/logs/DeepSeek-R1-Distill-Llama-8B_04020_2034_smt/training_matrix_sparse_.log" \
    2> "/ocean/projects/cis250057p/hhe4/LLM-FT/deepspeed/logs/DeepSeek-R1-Distill-Llama-8B_04020_2034_smt/err_matrix_.log"
```



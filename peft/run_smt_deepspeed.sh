conda activate "your-env"
cd "the-path-to-your-file"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=True
export CUDA_LAUNCH_BLOCKING=1

# deepspeed --include=localhost:0,1,2,3 train.py \
WANDB__SERVICE_WAIT=300
deepspeed --include=localhost:0,1,2,3 --master_port 60000 train_smt.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-7b-hf" \
--chat_template_format None \
--use_peft_smt True \
--smt_dropout 0.0 \
--smt_offload False \
--num_submatrix_mlp 0 \
--num_submatrix_attn 890 \
--selection_strategy "no_restriction" \
--calculation_strategy "mean_abs" \
--smt_learning_rate 1e-4 \
--smt_w_decay 0.0 \
--target_modules ['q_proj', 'k_proj', 'v_proj'] \
--dataset_name_or_path "path-to-dataset-commonsense_170k.json" \
--eval_set_size 120 \
--add_special_tokens False \
--append_concat_token False \
--packing False \
--max_seq_length 2048 \
--num_train_epochs 3 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_steps 30 \
--evaluation_strategy "steps" \
--save_strategy "epoch" \
--hub_private_repo False \
--hub_strategy "every_save" \
--bf16 True \
--learning_rate 9.65e-6 \
--lr_scheduler_type "linear" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "deepspeed-smt" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing True \
--use_reentrant False \
--full_ft_steps 100 \
--fft_offload True \
--fft_zero_stage 2 \
--load_best_model_at_end True \
--metric_for_best_model ""eval_loss"" \
--smt_deepspeed "/smt_deepspeed_config.json" \
--deepspeed "/fft_deepspeed_config.json"







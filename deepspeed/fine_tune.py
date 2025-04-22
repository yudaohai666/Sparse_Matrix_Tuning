import copy
import logging

from pathlib import Path
import torch.nn as nn
import gc
import tqdm
from collections import defaultdict
import functools
from dataclasses import dataclass
from typing import Dict, Sequence

from pytorch_memlab import MemReporter
from transformers import (
    SchedulerType,
    get_scheduler,
)
import torch
import re
import transformers
import json
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

# for step2_reward, need dschat from deepspeed example official website
# from dschat.utils.model.model_utils import create_critic_model
# from dschat.utils.data.data_utils import create_prompt_dataset, DataCollatorReward

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param, safe_get_full_optimizer_state, safe_set_full_optimizer_state
import os
from transformers import AutoModelForCausalLM, SchedulerType, get_scheduler
import time
import argparse
import math

from smt.smt import convert_linear_layer_to_matrix_sparsity, get_optimizer_sparse_grouped_parameters, get_optimizer_qk_augment_grouped_parameters, freeze_unselected_matrix_layer, freeze_unselected_channel_layer, convert_linear_layer_to_channel_sparsity
from smt.smt_helper import select_submatrix_based_on_grads, get_blocks, get_named_linears, select_channel_based_on_activation

from helpers.deepspeed_helpers import (
    get_train_ds_config,
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

from helpers.helper import (
    SupervisedDataset,
    DataCollatorForSupervisedDataset,
    evaluation,
    make_supervised_data_module,
    final_eval_save_model,
    epoch_save_model,
    iteration_save_model,
    save_pretrained_model,
    # parse_args,
)


# the trainer function refer to DeepSpeed Examples:
# Source Code please refer to:
# https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py
def trainer(args):
    # args = parse_args()
    if args.local_rank == -1:
        torch.autograd.Function
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # you can customize backend!!! mpi, gloo, nccl
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # torch.distributed.init_process_group("gloo", rank=args.global_rank, world_size=torch.distributed.get_world_size())
    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=False,
                                    tb_path="step1_tensorboard",
                                    profiler_path=args.output_dir)

    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_ft_batch_size
    ds_config['zero_allow_untested_optimizer'] = True
    ds_config[
        'train_batch_size'] = args.per_device_ft_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    ds_config['zero_force_ds_cpu_optimizer'] = False
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # Hector: only for step2 llama, use load_hf_tokenizer to build directly.
    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    # args.end_of_conversation_token = "<|endoftext|>"
    # additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    ##test
    logging.warning("Loading tokenizer...")
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  args.max_seq_len,
                                  fast_tokenizer=True)
    logging.warning("tokenizer Loaded...")
    logging.warning(f"tokenizer max_length:{tokenizer.model_max_length}")

    try:
        train_module_dict = make_supervised_data_module(tokenizer, args)
    except:
        raise ValueError(
            'Data load error, recheck data path, data format please')

    # the dataloader function refer to DeepSpeed Examples:
    # Source Code please refer to:
    # https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_module_dict["train_dataset"])
        eval_sampler = SequentialSampler(train_module_dict["eval_dataset"])
    else:
        train_sampler = DistributedSampler(train_module_dict["train_dataset"])
        eval_sampler = DistributedSampler(train_module_dict["eval_dataset"])

    train_dataloader = DataLoader(
        train_module_dict["train_dataset"],
        sampler=train_sampler,
        batch_size=args.per_device_ft_batch_size,
        collate_fn=train_module_dict["data_collator"],
    )
    eval_dataloader = DataLoader(
        train_module_dict["eval_dataset"],
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=train_module_dict["data_collator"],
    )

    print_rank_0("Loading model...")
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)
    print_rank_0(f"model loaded from {args.model_name_or_path} successfully!")

    # set the learning rate, wrap the parameters, and ready to start full fine-tuning
    # Configure the learning rate, prepare parameters, and set up for fine-tuning

    if args.qk_scheduler:
        optimizer_grouped_parameters = get_optimizer_qk_augment_grouped_parameters(
            model, args.w_decay, args.ft_learning_rate,
            args.ft_learning_rate * args.qk_lr_times)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.w_decay, args.ft_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.ft_learning_rate,
                              betas=(0.9, 0.95))

    del optimizer_grouped_parameters

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_ft_epochs * num_update_steps_per_epoch,
    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    model.gradient_checkpointing_enable()

    # Training
    print_rank_0("***** Running training *****", args.global_rank)
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    print_rank_0(formatted_args, args.global_rank)

    current_step_count = 0
    final_saved_model_index = 'final'

    warmup_grads = {}
    attention_warmup_grads = {}

    activation = {}
    attention_activation = {}

    targeted_module_dims = {}

    SMT_Convert = False
    Channel_Sparse_Convert = False

    best_eval_loss = float('inf')
    best_model = None

    TARGET_MODULE_NAMES = {
                    'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj',
                    'v_proj'
                }
    for name, param in model.module.named_parameters():
        if 'weight' in name:
            for target_module_name in TARGET_MODULE_NAMES:
                if target_module_name in name and target_module_name not in targeted_module_dims:
                    targeted_module_dims[target_module_name] = [
                        param.shape[0], param.shape[1]
                    ]
                    break
    print("targeted_module_dims", targeted_module_dims)

    num_total_blocks = 0
    for name, param in model.module.named_parameters():
        if isinstance(param, torch.Tensor) and param.ndim == 2:
            num_total_blocks += param.shape[0] / 256 * param.shape[1] / 256

    num_downsampled_attention_blocks = int(args.downsample_attention_blocks_ratio * num_total_blocks)
    print("Number of downsampled attention blocks:", num_downsampled_attention_blocks ,
          "with downsample_attention_blocks_ratio:", args.downsample_attention_blocks_ratio)
    num_downsampled_mlp_blocks = int(args.downsample_mlp_blocks_ratio * num_total_blocks)
    print("Number of mlp attention blocks:", num_downsampled_mlp_blocks ,
          "with downsample_mlp_blocks_ratio:", args.downsample_mlp_blocks_ratio)


    # save_pretrained_model(model, tokenizer, args)
    training_loss_list = []
    eval_loss_list = []
    ppl_list = []
    for epoch in range(args.num_ft_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_ft_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        mean_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            ################################### SMT Selection Start #################################
            ## Matrix Sparsity selection!!! select_submatrix_based_on_grads()
            if args.matrix_sparsity and current_step_count == args.full_ft_steps and not SMT_Convert:

                # grads (module_name, layer_number, head_number): value
                # Hector's Notes: select the top n sub-matrix based on the mean().abs() value of the gradients
                # Hector's Notes: warmup grads have been accumulated through all warm up iterations
                # initialize
                SMT_Convert = True
                selected_submatrix_attention = {}
                selected_submatrix = {}

                if args.no_limit_mixture:
                    print_rank_0(
                        f"apply matrix sparsity to both attention and mlp without any limitation",
                        args.global_rank)
                    attention_warmup_grads.update(warmup_grads)
                    selected_submatrix = select_submatrix_based_on_grads(
                        warmup_grads,
                        targeted_module_dims,
                        num_downsampled_mlp_blocks + num_downsampled_attention_blocks,
                        selection_strategy=args.selection_strategy,
                        calculate_strategy=args.calculate_strategy,
                        model=args.model_name_or_path,
                        do_gradient_distribution_analysis=args.do_gradient_distribution_analysis,
                        output_dir=args.output_dir)
                    # set the freeze for the unselected heads
                    # TODO: Need to revise freeze function
                    model = freeze_unselected_matrix_layer(model.module,
                                                           selected_submatrix,
                                                           {},
                                                           mixture=True)
                else:
                    if num_downsampled_attention_blocks > 0:
                        print_rank_0(
                            f"apply matrix sparsity to attention layer",
                            args.global_rank)
                        # print_rank_0(
                        #     f"attention warmup grads{attention_warmup_grads}",
                        #     args.global_rank)

                        # To get the accumulation gradient for Q, K, V vectors in every single layer.
                        attention_accumulated_grads = {}
                        for key, value in attention_warmup_grads.items():
                            attention_accumulated_grads[key] = torch.mean(
                                torch.abs(attention_warmup_grads[key]))

                        # print_rank_0(
                        #     f"attention warmup grads information(magnitude of avg(abs())) for Q, K, V{attention_accumulated_grads}",
                        #     args.global_rank)

                        selected_submatrix_attention = select_submatrix_based_on_grads(
                            attention_warmup_grads,
                            targeted_module_dims,
                            num_downsampled_attention_blocks,
                            selection_strategy=args.selection_strategy,
                            model=args.model_name_or_path,
                            do_gradient_distribution_analysis=args.do_gradient_distribution_analysis,
                            output_dir=args.output_dir)
                        # print_rank_0(
                        #     f"selected attention submatrix: {selected_submatrix_attention}",
                        #     args.global_rank)

                    if num_downsampled_mlp_blocks > 0:
                        selected_submatrix = select_submatrix_based_on_grads(
                            warmup_grads,
                            targeted_module_dims,
                            num_downsampled_mlp_blocks,
                            selection_strategy=args.selection_strategy,
                            calculate_strategy=args.calculate_strategy,
                            model=args.model_name_or_path,
                            do_gradient_distribution_analysis=args.do_gradient_distribution_analysis,
                            output_dir=args.output_dir)
                        # print_rank_0(
                        #     f"selected mlp submatrix: {selected_submatrix}",
                        #     args.global_rank)
                        # set the freeze for the unselected heads
                        # TODO: Need to revise freeze function

                    model = freeze_unselected_matrix_layer(
                        model.module, selected_submatrix,
                        selected_submatrix_attention)

                # convert selected mlp/attention to linear_matrix_sparsity
                print(
                    "===========================SYSTEM IMPLEMENTATION======================================"
                )
                model = convert_linear_layer_to_matrix_sparsity(
                    model, selected_submatrix, selected_submatrix_attention)

                model = make_model_gradient_checkpointing_compatible(model)

                optimizer_grouped_parameters = get_optimizer_sparse_grouped_parameters(
                    model, args.w_decay, args.smt_lr)

                # AdamSparseOptimizer = DeepSpeedCPUMatrixSparisityAdam if args.offload else FusedMatrixSparseAdam
                # AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
                AdamOptimizer = FusedAdam

                del optimizer
                del lr_scheduler
                del warmup_grads
                del attention_warmup_grads

                # in SMT paper, all results are obtained under ft_learning_rate = smt_lr = 9.65e-6.
                # Further experiments: set smt_lr = 1e-4 can further improve the accuracy.
                new_optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                              lr=args.ft_learning_rate,
                                              betas=(0.9, 0.95))

                del optimizer_grouped_parameters

                new_lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=new_optimizer,
                    num_warmup_steps=args.smt_lr_warmup_steps,
                    num_training_steps=args.num_ft_epochs *
                    num_update_steps_per_epoch - current_step_count,
                )

                ds_config["zero_optimization"]["offload_param"][
                    "device"] = "none"
                ds_config["zero_optimization"]["offload_optimizer"][
                    "device"] = "none"
                model, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=model,
                    optimizer=new_optimizer,
                    args=args,
                    config=ds_config,
                    lr_scheduler=new_lr_scheduler)

                print("Print the requres_grad of model weight.")
                for name, param in model.module.named_parameters():
                    print(name, "requres_grad", param.requires_grad)

                # print matrix sparsity trainable parameters
                total_num = sum(p.numel() for p in model.module.parameters())
                selected_num = sum(p.numel()
                                   for p in model.module.parameters()
                                   if p.requires_grad)
                print_rank_0(f"Number of Total parameters: {total_num}",
                             args.global_rank)
                rate = (selected_num / total_num) * 100
                print_rank_0(
                    f"Number of trainable parameters: {selected_num}, about{rate}% matrix sparsity parameters in the model are training",
                    args.global_rank)
                # break
            ################################### SMT Selection finish #################################

            ################################### Activation Selection Start #################################
            ## Matrix Sparsity selection!!! select_submatrix_based_on_grads()
            if args.channel_sparsity and current_step_count == args.full_ft_steps and not Channel_Sparse_Convert:

                # grads (module_name, layer_number, head_number): value
                # Hector's Notes: select the top n sub-matrix based on the mean().abs() value of the gradients
                # Hector's Notes: warmup grads have been accumulated through all warm up iterations

                # function need to revise: 1. select_submatrix_based_on_grads -> select_channel_based_on_activation
                # 2. freeze_unselected_matrix_layer -> freeze_unselected_channel_layer
                # 3. convert_linear_layer_to_matrix_sparsity -> convert_linear_layer_to_channel_sparsity

                # initialize
                Channel_Sparse_Convert = True
                selected_channel_attention = {}
                selected_channel = {}

                if args.no_limit_mixture:
                    print_rank_0(
                        f"apply matrix sparsity to both attention and mlp without any limitation",
                        args.global_rank)

                    # if torch.distributed.get_world_size() > 0:
                    #     activation.to(device)
                    #     for key, value in activation.items():
                    #         # torch.distributed with nccl backend only support gpu communication.
                    #         deepspeed.comm.all_reduce(activation[key])
                    #         # torch.distributed.all_reduce(activation[key], op=torch.distributed.ReduceOp.SUM)
                    #     activation.cpu()
                    #
                    #     attention_activation.to(device)
                    #     for key, value in attention_activation.items():
                    #         deepspeed.comm.all_reduce(attention_activation[key])
                    #         # torch.distributed.all_reduce(attention_activation[key], op=torch.distributed.ReduceOp.SUM)
                    #     attention_activation.cpu()

                    attention_activation.update(activation)
                    selected_channel = select_channel_based_on_activation(
                        activation,
                        args.num_attention_channel + args.num_mlp_channel,
                        selection_strategy=args.selection_strategy,
                        calculate_strategy=args.calculate_strategy,
                        model=args.model_name_or_path)
                    # set the freeze for the unselected heads
                    # TODO: Need to revise freeze function
                    model = freeze_unselected_channel_layer(model.module,
                                                            selected_channel,
                                                            {},
                                                            mixture=True)
                else:
                    if args.num_attention_channel > 0:
                        # if torch.distributed.get_world_size() > 1:
                        #     attention_activation.to(device)
                        #     for key, value in attention_activation.items():
                        #         # torch.distributed with nccl backend only support gpu communication.
                        #         deepspeed.comm.all_reduce(attention_activation[key])
                        #         # torch.distributed.all_reduce(attention_activation[key], op=torch.distributed.ReduceOp.SUM)
                        #     attention_activation.cpu()

                        # same `selected_channel_attention` tested: pass.
                        # selected_channel_attention = {('v_proj', 31): [2533, 1415, 3431, 2789, 1512, 3209, 2393, 1076, 2158, 2235, 1404, 257, 2298, 363, 490, 1793, 94, 2608, 4071, 3135, 2469, 588, 2042, 2927, 462, 339, 2883, 2350, 2622, 3571, 580, 1218, 1360, 1571, 1927, 3994, 4076, 1839, 1583, 2281, 1335, 575, 3971, 3700, 3591, 3656, 3257, 2459, 3241], ('q_proj', 31): [2533, 1415, 3431, 2789, 1512, 3209, 2393, 1076, 2158, 2235, 1404, 257, 2298, 363, 490, 1793, 94, 2608, 4071, 3135, 2469, 588, 2042, 2927, 462, 339, 2883, 2350, 2622, 3571, 580, 1218, 1360, 1571, 1927, 3994, 4076, 1839, 1583, 2281, 1335, 575, 3971, 3700, 3591, 3656, 3257, 2459, 3241], ('k_proj', 31): [2533, 1415, 3431, 2789, 1512, 3209, 2393, 1076, 2158, 2235, 1404, 257, 2298, 363, 490, 1793, 94, 2608, 4071, 3135, 2469, 588, 2042, 2927, 462, 339, 2883, 2350, 2622, 3571, 580, 1218, 1360, 1571, 1927, 3994, 4076, 1839, 1583, 2281, 1335, 575, 3971, 3700, 3591, 3656, 3257, 2459, 3241], ('v_proj', 1): [1512, 310, 2298, 2944, 216, 2393, 2544, 3135, 3443, 1415, 3893, 1218, 2608, 2469, 2235, 490, 2158, 3209, 1076, 2147, 2533, 3807, 1411, 2077, 759, 2789, 257, 3994, 3946, 914, 2958, 2189, 363, 3431, 1054, 580, 375, 2345, 3934, 3164, 1270, 812, 3852, 2649], ('q_proj', 1): [1512, 310, 2298, 2944, 216, 2393, 2544, 3135, 3443, 1415, 3893, 1218, 2608, 2469, 2235, 490, 2158, 3209, 1076, 2147, 2533, 3807, 1411, 2077, 759, 2789, 257, 3994, 3946, 914, 2958, 2189, 363, 3431, 1054, 580, 375, 2345, 3934, 3164, 1270, 812, 3852, 2649], ('k_proj', 1): [1512, 310, 2298, 2944, 216, 2393, 2544, 3135, 3443, 1415, 3893, 1218, 2608, 2469, 2235, 490, 2158, 3209, 1076, 2147, 2533, 3807, 1411, 2077, 759, 2789, 257, 3994, 3946, 914, 2958, 2189, 363, 3431, 1054, 580, 375, 2345, 3934, 3164, 1270, 812, 3852, 2649], ('v_proj', 11): [2533, 1076, 2789, 3431, 339, 2158, 2350, 2393, 3209, 2298, 257, 490, 4071, 2235, 94, 2469, 1404, 3994, 3135, 1512, 363, 2608, 1218, 462, 2927, 2168, 3999, 1571, 1793, 588, 2459, 3656, 2863, 2944, 1272, 2016, 3893, 788, 1583, 2050, 474, 575, 2622, 3471, 3491, 3872, 2958, 580, 2147], ('q_proj', 11): [2533, 1076, 2789, 3431, 339, 2158, 2350, 2393, 3209, 2298, 257, 490, 4071, 2235, 94, 2469, 1404, 3994, 3135, 1512, 363, 2608, 1218, 462, 2927, 2168, 3999, 1571, 1793, 588, 2459, 3656, 2863, 2944, 1272, 2016, 3893, 788, 1583, 2050, 474, 575, 2622, 3471, 3491, 3872, 2958, 580, 2147], ('k_proj', 11): [2533, 1076, 2789, 3431, 339, 2158, 2350, 2393, 3209, 2298, 257, 490, 4071, 2235, 94, 2469, 1404, 3994, 3135, 1512, 363, 2608, 1218, 462, 2927, 2168, 3999, 1571, 1793, 588, 2459, 3656, 2863, 2944, 1272, 2016, 3893, 788, 1583, 2050, 474, 575, 2622, 3471, 3491, 3872, 2958, 580, 2147], ('v_proj', 26): [2533, 3431, 1076, 2789, 3209, 2158, 2298, 2235, 1512, 2393, 3135, 4071, 363, 1404, 257, 2944, 3471, 575, 3994, 462, 580, 2608, 2469, 1793, 339, 3893, 2350, 1218, 2622, 490, 2883, 3656, 2958, 2927, 788, 94, 1571, 2050, 2077, 2863, 1307, 3946, 588, 2459, 3852, 310, 375, 1524, 239, 2281, 972, 2611, 3391, 1825, 4063, 3148, 2358, 1721, 2403, 3536, 2168, 1163, 2016, 1868, 1523, 866, 2625, 1117, 1167, 3117, 707, 886, 2018, 1548, 2627, 3290, 3807, 3771, 3921, 1449, 1316, 2476, 2540, 1734, 2573, 1583, 1888, 1510, 3106, 1272, 1057, 3952, 3629, 470, 2226, 3556, 134, 200, 2825, 3772, 2248, 2815, 1360, 299, 3270, 448, 1930, 3819, 3065, 1062, 3386, 1119, 2410, 1399, 2282, 2140, 1334, 1637, 3722, 309, 1130, 2750, 1409, 2199, 2095, 3999, 2688, 3080, 1926, 955, 2204, 358, 1501, 3249, 1354, 1374, 2377, 2551, 1257, 669, 3445, 3491, 2543, 4082, 1261, 1492, 556, 1134, 1857, 3514, 1944, 1146, 2026, 3579, 568, 3767, 3245, 1620, 739, 3362, 1090, 2654, 805, 252, 3526, 587, 1441, 3816, 3866, 2736, 2968, 2070, 1619, 2205, 3205, 2187, 750, 803, 367, 2881, 2326, 685, 3151, 3168, 2272, 189, 3840, 2869, 662, 3577, 3662, 3803, 2718, 3051, 2581, 2433, 3506, 1461, 2131, 2048, 102, 3731, 2520, 2949, 6, 1656, 3324, 2513, 3854, 1017, 2038, 368, 2801, 700, 1269, 3872, 2036, 2896, 4047, 2147, 2365, 1180, 2290, 513, 3512, 3427, 2924, 2203, 883, 809, 1755, 2578, 3075, 745, 2572, 45, 1481, 3169, 1636, 1113, 2022, 2705, 3619, 3562, 1051, 4085, 1007, 2333, 1661, 188, 759, 3830, 3663, 3449, 3678, 3233, 3814, 814, 666, 1045, 1489, 2279, 66, 3674, 3529, 2412, 2612, 487, 1841, 3710, 1746, 1551, 729, 1174, 3836, 3979, 2114, 2579, 3067, 2952, 1545, 1519, 1053, 181, 3253, 884, 1648, 1378, 2633, 1972, 3348, 3173, 371, 1478, 2993, 4055, 3863, 2141, 956, 2522, 1303, 3241, 1012, 968, 3333, 949, 1473, 1249, 3397, 2643, 3123, 911, 3490, 2254, 443, 2786, 747, 3981, 2852, 3172, 1371, 372, 2544, 1227, 412, 158, 138, 2928, 2913, 3194, 2160, 3837, 505, 1686], ('q_proj', 26): [2533, 3431, 1076, 2789, 3209, 2158, 2298, 2235, 1512, 2393, 3135, 4071, 363, 1404, 257, 2944, 3471, 575, 3994, 462, 580, 2608, 2469, 1793, 339, 3893, 2350, 1218, 2622, 490, 2883, 3656, 2958, 2927, 788, 94, 1571, 2050, 2077, 2863, 1307, 3946, 588, 2459, 3852, 310, 375, 1524, 239, 2281, 972, 2611, 3391, 1825, 4063, 3148, 2358, 1721, 2403, 3536, 2168, 1163, 2016, 1868, 1523, 866, 2625, 1117, 1167, 3117, 707, 886, 2018, 1548, 2627, 3290, 3807, 3771, 3921, 1449, 1316, 2476, 2540, 1734, 2573, 1583, 1888, 1510, 3106, 1272, 1057, 3952, 3629, 470, 2226, 3556, 134, 200, 2825, 3772, 2248, 2815, 1360, 299, 3270, 448, 1930, 3819, 3065, 1062, 3386, 1119, 2410, 1399, 2282, 2140, 1334, 1637, 3722, 309, 1130, 2750, 1409, 2199, 2095, 3999, 2688, 3080, 1926, 955, 2204, 358, 1501, 3249, 1354, 1374, 2377, 2551, 1257, 669, 3445, 3491, 2543, 4082, 1261, 1492, 556, 1134, 1857, 3514, 1944, 1146, 2026, 3579, 568, 3767, 3245, 1620, 739, 3362, 1090, 2654, 805, 252, 3526, 587, 1441, 3816, 3866, 2736, 2968, 2070, 1619, 2205, 3205, 2187, 750, 803, 367, 2881, 2326, 685, 3151, 3168, 2272, 189, 3840, 2869, 662, 3577, 3662, 3803, 2718, 3051, 2581, 2433, 3506, 1461, 2131, 2048, 102, 3731, 2520, 2949, 6, 1656, 3324, 2513, 3854, 1017, 2038, 368, 2801, 700, 1269, 3872, 2036, 2896, 4047, 2147, 2365, 1180, 2290, 513, 3512, 3427, 2924, 2203, 883, 809, 1755, 2578, 3075, 745, 2572, 45, 1481, 3169, 1636, 1113, 2022, 2705, 3619, 3562, 1051, 4085, 1007, 2333, 1661, 188, 759, 3830, 3663, 3449, 3678, 3233, 3814, 814, 666, 1045, 1489, 2279, 66, 3674, 3529, 2412, 2612, 487, 1841, 3710, 1746, 1551, 729, 1174, 3836, 3979, 2114, 2579, 3067, 2952, 1545, 1519, 1053, 181, 3253, 884, 1648, 1378, 2633, 1972, 3348, 3173, 371, 1478, 2993, 4055, 3863, 2141, 956, 2522, 1303, 3241, 1012, 968, 3333, 949, 1473, 1249, 3397, 2643, 3123, 911, 3490, 2254, 443, 2786, 747, 3981, 2852, 3172, 1371, 372, 2544, 1227, 412, 158, 138, 2928, 2913, 3194, 2160, 3837, 505, 1686], ('k_proj', 26): [2533, 3431, 1076, 2789, 3209, 2158, 2298, 2235, 1512, 2393, 3135, 4071, 363, 1404, 257, 2944, 3471, 575, 3994, 462, 580, 2608, 2469, 1793, 339, 3893, 2350, 1218, 2622, 490, 2883, 3656, 2958, 2927, 788, 94, 1571, 2050, 2077, 2863, 1307, 3946, 588, 2459, 3852, 310, 375, 1524, 239, 2281, 972, 2611, 3391, 1825, 4063, 3148, 2358, 1721, 2403, 3536, 2168, 1163, 2016, 1868, 1523, 866, 2625, 1117, 1167, 3117, 707, 886, 2018, 1548, 2627, 3290, 3807, 3771, 3921, 1449, 1316, 2476, 2540, 1734, 2573, 1583, 1888, 1510, 3106, 1272, 1057, 3952, 3629, 470, 2226, 3556, 134, 200, 2825, 3772, 2248, 2815, 1360, 299, 3270, 448, 1930, 3819, 3065, 1062, 3386, 1119, 2410, 1399, 2282, 2140, 1334, 1637, 3722, 309, 1130, 2750, 1409, 2199, 2095, 3999, 2688, 3080, 1926, 955, 2204, 358, 1501, 3249, 1354, 1374, 2377, 2551, 1257, 669, 3445, 3491, 2543, 4082, 1261, 1492, 556, 1134, 1857, 3514, 1944, 1146, 2026, 3579, 568, 3767, 3245, 1620, 739, 3362, 1090, 2654, 805, 252, 3526, 587, 1441, 3816, 3866, 2736, 2968, 2070, 1619, 2205, 3205, 2187, 750, 803, 367, 2881, 2326, 685, 3151, 3168, 2272, 189, 3840, 2869, 662, 3577, 3662, 3803, 2718, 3051, 2581, 2433, 3506, 1461, 2131, 2048, 102, 3731, 2520, 2949, 6, 1656, 3324, 2513, 3854, 1017, 2038, 368, 2801, 700, 1269, 3872, 2036, 2896, 4047, 2147, 2365, 1180, 2290, 513, 3512, 3427, 2924, 2203, 883, 809, 1755, 2578, 3075, 745, 2572, 45, 1481, 3169, 1636, 1113, 2022, 2705, 3619, 3562, 1051, 4085, 1007, 2333, 1661, 188, 759, 3830, 3663, 3449, 3678, 3233, 3814, 814, 666, 1045, 1489, 2279, 66, 3674, 3529, 2412, 2612, 487, 1841, 3710, 1746, 1551, 729, 1174, 3836, 3979, 2114, 2579, 3067, 2952, 1545, 1519, 1053, 181, 3253, 884, 1648, 1378, 2633, 1972, 3348, 3173, 371, 1478, 2993, 4055, 3863, 2141, 956, 2522, 1303, 3241, 1012, 968, 3333, 949, 1473, 1249, 3397, 2643, 3123, 911, 3490, 2254, 443, 2786, 747, 3981, 2852, 3172, 1371, 372, 2544, 1227, 412, 158, 138, 2928, 2913, 3194, 2160, 3837, 505, 1686], ('v_proj', 25): [2533, 3431, 2789, 1076, 3209, 2158, 2298, 2235, 1512, 3135, 2393, 4071, 363, 2944, 1404, 3471, 575, 257, 3994, 462, 580, 2883, 3893, 1793, 2608, 339, 2350, 1218, 2958, 2469, 490, 2927, 2622, 94, 3656, 1571, 788, 2863, 2050, 2077, 3946, 1307, 3852, 375, 2459, 588, 239, 1524, 310, 1163, 2281, 2358, 2611, 3117, 1523, 1057, 1868, 2168, 972, 1721, 4063, 3536, 707, 2016, 1825, 3148, 1489, 3290, 2248, 1548, 2573, 3249, 1316, 1378, 1167, 1449, 2204, 200, 3391, 685, 299, 1583, 3819, 487, 886, 3491, 2825, 2199, 2736, 739, 134, 3840, 866, 3514, 3772, 2949, 1117, 2540, 1734, 1930, 3807, 2476, 2403, 2522, 2815, 3333, 3526, 2688, 3674, 2226, 3106, 3921, 2282, 1545, 3270, 1409, 2750, 448, 3722, 470, 1090, 2718, 2643, 2070, 3731, 2433, 1492, 1062, 2869, 2095, 1360, 1374, 3065, 2581, 2187, 3629, 712, 1130, 1461, 2026, 3562, 4085, 1134, 1119, 3952, 2377, 3362, 1272, 2625, 1510, 1180, 3662, 3568, 556, 3872, 3245, 1334, 309, 3836, 3771, 378, 1399, 3205, 3490, 1592, 368, 2140, 1857, 2205, 1888, 1620, 1053, 1113, 584, 3678, 2513, 2018, 2801, 66, 1174, 3051, 3814, 2627, 150, 3579, 345, 745, 2998, 1885, 2131, 4048, 2687, 45, 1354, 5, 3999, 3273, 3854, 3866, 786, 358, 1551, 3803, 1661, 1636, 7, 803, 3767, 956, 949, 3556, 3636, 3506, 1877, 2192, 1619, 1944, 4047, 780, 3371, 2959, 1972, 1478, 2038, 3168, 955, 587, 2410, 2147, 3816, 2654, 655, 3, 2742, 729, 4082, 644, 856, 1045, 1926, 2579, 750, 968, 3427, 1690, 2022, 2329, 3386, 2544, 1861, 999, 2543, 1501, 2326, 1233, 261, 2551, 1249, 3173, 2612, 4076, 3110, 1772, 3703, 2310, 2572, 338, 372, 1664, 1841, 2974, 3863, 3080, 2968, 2441, 1479, 3151, 1025, 1648, 292, 116, 1012, 2582, 2447, 2520, 186, 2952, 3818, 689, 2272, 446, 3031, 2588, 367, 51, 3689, 3066, 3922, 1656, 2195, 1520, 3687, 1998, 1312, 1151, 1227, 1051, 1441, 3979, 4077, 1073, 1392, 988, 102, 1637, 3123, 3445, 2114, 2947, 2333, 4055, 4053, 513, 6, 1768, 1269, 1755, 2765, 3542, 851, 1007, 985, 2737, 1472, 1519, 1003, 2864, 3348, 1539, 805, 3262, 3512, 1569, 3067, 2160, 2786, 2740, 934, 2680, 2181, 252, 3449, 1257, 3942, 2365, 3075, 347, 3825, 1763, 3700, 596, 1371, 2798, 304, 2308, 814], ('q_proj', 25): [2533, 3431, 2789, 1076, 3209, 2158, 2298, 2235, 1512, 3135, 2393, 4071, 363, 2944, 1404, 3471, 575, 257, 3994, 462, 580, 2883, 3893, 1793, 2608, 339, 2350, 1218, 2958, 2469, 490, 2927, 2622, 94, 3656, 1571, 788, 2863, 2050, 2077, 3946, 1307, 3852, 375, 2459, 588, 239, 1524, 310, 1163, 2281, 2358, 2611, 3117, 1523, 1057, 1868, 2168, 972, 1721, 4063, 3536, 707, 2016, 1825, 3148, 1489, 3290, 2248, 1548, 2573, 3249, 1316, 1378, 1167, 1449, 2204, 200, 3391, 685, 299, 1583, 3819, 487, 886, 3491, 2825, 2199, 2736, 739, 134, 3840, 866, 3514, 3772, 2949, 1117, 2540, 1734, 1930, 3807, 2476, 2403, 2522, 2815, 3333, 3526, 2688, 3674, 2226, 3106, 3921, 2282, 1545, 3270, 1409, 2750, 448, 3722, 470, 1090, 2718, 2643, 2070, 3731, 2433, 1492, 1062, 2869, 2095, 1360, 1374, 3065, 2581, 2187, 3629, 712, 1130, 1461, 2026, 3562, 4085, 1134, 1119, 3952, 2377, 3362, 1272, 2625, 1510, 1180, 3662, 3568, 556, 3872, 3245, 1334, 309, 3836, 3771, 378, 1399, 3205, 3490, 1592, 368, 2140, 1857, 2205, 1888, 1620, 1053, 1113, 584, 3678, 2513, 2018, 2801, 66, 1174, 3051, 3814, 2627, 150, 3579, 345, 745, 2998, 1885, 2131, 4048, 2687, 45, 1354, 5, 3999, 3273, 3854, 3866, 786, 358, 1551, 3803, 1661, 1636, 7, 803, 3767, 956, 949, 3556, 3636, 3506, 1877, 2192, 1619, 1944, 4047, 780, 3371, 2959, 1972, 1478, 2038, 3168, 955, 587, 2410, 2147, 3816, 2654, 655, 3, 2742, 729, 4082, 644, 856, 1045, 1926, 2579, 750, 968, 3427, 1690, 2022, 2329, 3386, 2544, 1861, 999, 2543, 1501, 2326, 1233, 261, 2551, 1249, 3173, 2612, 4076, 3110, 1772, 3703, 2310, 2572, 338, 372, 1664, 1841, 2974, 3863, 3080, 2968, 2441, 1479, 3151, 1025, 1648, 292, 116, 1012, 2582, 2447, 2520, 186, 2952, 3818, 689, 2272, 446, 3031, 2588, 367, 51, 3689, 3066, 3922, 1656, 2195, 1520, 3687, 1998, 1312, 1151, 1227, 1051, 1441, 3979, 4077, 1073, 1392, 988, 102, 1637, 3123, 3445, 2114, 2947, 2333, 4055, 4053, 513, 6, 1768, 1269, 1755, 2765, 3542, 851, 1007, 985, 2737, 1472, 1519, 1003, 2864, 3348, 1539, 805, 3262, 3512, 1569, 3067, 2160, 2786, 2740, 934, 2680, 2181, 252, 3449, 1257, 3942, 2365, 3075, 347, 3825, 1763, 3700, 596, 1371, 2798, 304, 2308, 814], ('k_proj', 25): [2533, 3431, 2789, 1076, 3209, 2158, 2298, 2235, 1512, 3135, 2393, 4071, 363, 2944, 1404, 3471, 575, 257, 3994, 462, 580, 2883, 3893, 1793, 2608, 339, 2350, 1218, 2958, 2469, 490, 2927, 2622, 94, 3656, 1571, 788, 2863, 2050, 2077, 3946, 1307, 3852, 375, 2459, 588, 239, 1524, 310, 1163, 2281, 2358, 2611, 3117, 1523, 1057, 1868, 2168, 972, 1721, 4063, 3536, 707, 2016, 1825, 3148, 1489, 3290, 2248, 1548, 2573, 3249, 1316, 1378, 1167, 1449, 2204, 200, 3391, 685, 299, 1583, 3819, 487, 886, 3491, 2825, 2199, 2736, 739, 134, 3840, 866, 3514, 3772, 2949, 1117, 2540, 1734, 1930, 3807, 2476, 2403, 2522, 2815, 3333, 3526, 2688, 3674, 2226, 3106, 3921, 2282, 1545, 3270, 1409, 2750, 448, 3722, 470, 1090, 2718, 2643, 2070, 3731, 2433, 1492, 1062, 2869, 2095, 1360, 1374, 3065, 2581, 2187, 3629, 712, 1130, 1461, 2026, 3562, 4085, 1134, 1119, 3952, 2377, 3362, 1272, 2625, 1510, 1180, 3662, 3568, 556, 3872, 3245, 1334, 309, 3836, 3771, 378, 1399, 3205, 3490, 1592, 368, 2140, 1857, 2205, 1888, 1620, 1053, 1113, 584, 3678, 2513, 2018, 2801, 66, 1174, 3051, 3814, 2627, 150, 3579, 345, 745, 2998, 1885, 2131, 4048, 2687, 45, 1354, 5, 3999, 3273, 3854, 3866, 786, 358, 1551, 3803, 1661, 1636, 7, 803, 3767, 956, 949, 3556, 3636, 3506, 1877, 2192, 1619, 1944, 4047, 780, 3371, 2959, 1972, 1478, 2038, 3168, 955, 587, 2410, 2147, 3816, 2654, 655, 3, 2742, 729, 4082, 644, 856, 1045, 1926, 2579, 750, 968, 3427, 1690, 2022, 2329, 3386, 2544, 1861, 999, 2543, 1501, 2326, 1233, 261, 2551, 1249, 3173, 2612, 4076, 3110, 1772, 3703, 2310, 2572, 338, 372, 1664, 1841, 2974, 3863, 3080, 2968, 2441, 1479, 3151, 1025, 1648, 292, 116, 1012, 2582, 2447, 2520, 186, 2952, 3818, 689, 2272, 446, 3031, 2588, 367, 51, 3689, 3066, 3922, 1656, 2195, 1520, 3687, 1998, 1312, 1151, 1227, 1051, 1441, 3979, 4077, 1073, 1392, 988, 102, 1637, 3123, 3445, 2114, 2947, 2333, 4055, 4053, 513, 6, 1768, 1269, 1755, 2765, 3542, 851, 1007, 985, 2737, 1472, 1519, 1003, 2864, 3348, 1539, 805, 3262, 3512, 1569, 3067, 2160, 2786, 2740, 934, 2680, 2181, 252, 3449, 1257, 3942, 2365, 3075, 347, 3825, 1763, 3700, 596, 1371, 2798, 304, 2308, 814], ('v_proj', 14): [2533, 1076, 3431, 2789, 3209, 2235, 2158, 2298, 2393, 339, 2350, 490, 3135, 1404, 2469, 4071, 1512, 257, 3471, 3994, 94, 2927, 363, 2863, 2944, 1218, 462, 575, 2608, 1793, 1571, 580, 588, 3893, 3656, 3872, 3999, 2168, 2958, 788, 1272, 2622, 2459, 108, 3852, 2050, 2036, 2883, 474, 3491, 1524, 975, 310, 2077, 1092], ('q_proj', 14): [2533, 1076, 3431, 2789, 3209, 2235, 2158, 2298, 2393, 339, 2350, 490, 3135, 1404, 2469, 4071, 1512, 257, 3471, 3994, 94, 2927, 363, 2863, 2944, 1218, 462, 575, 2608, 1793, 1571, 580, 588, 3893, 3656, 3872, 3999, 2168, 2958, 788, 1272, 2622, 2459, 108, 3852, 2050, 2036, 2883, 474, 3491, 1524, 975, 310, 2077, 1092], ('k_proj', 14): [2533, 1076, 3431, 2789, 3209, 2235, 2158, 2298, 2393, 339, 2350, 490, 3135, 1404, 2469, 4071, 1512, 257, 3471, 3994, 94, 2927, 363, 2863, 2944, 1218, 462, 575, 2608, 1793, 1571, 580, 588, 3893, 3656, 3872, 3999, 2168, 2958, 788, 1272, 2622, 2459, 108, 3852, 2050, 2036, 2883, 474, 3491, 1524, 975, 310, 2077, 1092], ('v_proj', 10): [2533, 1076, 3431, 2789, 339, 2158, 2350, 2393, 3209, 2298, 257, 4071, 490, 2235, 1512, 1404, 94, 2469, 3994, 3135, 363, 2608, 2168, 1218, 1571, 462, 3999, 1793, 2927, 588, 3656, 2944, 2459, 2016, 2863, 788, 3893, 1583, 2050, 2958, 2622, 575, 1272, 310, 2147, 580], ('q_proj', 10): [2533, 1076, 3431, 2789, 339, 2158, 2350, 2393, 3209, 2298, 257, 4071, 490, 2235, 1512, 1404, 94, 2469, 3994, 3135, 363, 2608, 2168, 1218, 1571, 462, 3999, 1793, 2927, 588, 3656, 2944, 2459, 2016, 2863, 788, 3893, 1583, 2050, 2958, 2622, 575, 1272, 310, 2147, 580], ('k_proj', 10): [2533, 1076, 3431, 2789, 339, 2158, 2350, 2393, 3209, 2298, 257, 4071, 490, 2235, 1512, 1404, 94, 2469, 3994, 3135, 363, 2608, 2168, 1218, 1571, 462, 3999, 1793, 2927, 588, 3656, 2944, 2459, 2016, 2863, 788, 3893, 1583, 2050, 2958, 2622, 575, 1272, 310, 2147, 580], ('v_proj', 12): [2533, 2789, 1076, 3431, 2158, 2393, 339, 2350, 3209, 2298, 2235, 257, 490, 94, 2469, 4071, 1404, 1512, 2608, 3994, 2927, 1218, 462, 363, 1571, 3135, 3999, 1793, 2863, 2168, 588, 2459, 3656, 1272, 2944, 3471, 788, 474, 1583, 2622, 2050, 3893, 3872, 3491, 2016, 2958, 575, 580, 2036], ('q_proj', 12): [2533, 2789, 1076, 3431, 2158, 2393, 339, 2350, 3209, 2298, 2235, 257, 490, 94, 2469, 4071, 1404, 1512, 2608, 3994, 2927, 1218, 462, 363, 1571, 3135, 3999, 1793, 2863, 2168, 588, 2459, 3656, 1272, 2944, 3471, 788, 474, 1583, 2622, 2050, 3893, 3872, 3491, 2016, 2958, 575, 580, 2036], ('k_proj', 12): [2533, 2789, 1076, 3431, 2158, 2393, 339, 2350, 3209, 2298, 2235, 257, 490, 94, 2469, 4071, 1404, 1512, 2608, 3994, 2927, 1218, 462, 363, 1571, 3135, 3999, 1793, 2863, 2168, 588, 2459, 3656, 1272, 2944, 3471, 788, 474, 1583, 2622, 2050, 3893, 3872, 3491, 2016, 2958, 575, 580, 2036], ('v_proj', 15): [2533, 3431, 1076, 2789, 2235, 3209, 2158, 2298, 2393, 339, 1512, 490, 3135, 2350, 3471, 2469, 1404, 257, 4071, 2927, 94, 3994, 1218, 2863, 2944, 363, 2608, 462, 575, 1571, 1793, 1415, 580, 588, 3656, 3893, 3872, 2168, 2958, 788, 2622, 2459, 3999, 1272, 3852, 2050, 3491, 108, 2883, 474, 2036, 1092, 1524, 975, 1307, 2077, 310, 3946, 3979], ('q_proj', 15): [2533, 3431, 1076, 2789, 2235, 3209, 2158, 2298, 2393, 339, 1512, 490, 3135, 2350, 3471, 2469, 1404, 257, 4071, 2927, 94, 3994, 1218, 2863, 2944, 363, 2608, 462, 575, 1571, 1793, 1415, 580, 588, 3656, 3893, 3872, 2168, 2958, 788, 2622, 2459, 3999, 1272, 3852, 2050, 3491, 108, 2883, 474, 2036, 1092, 1524, 975, 1307, 2077, 310, 3946, 3979], ('k_proj', 15): [2533, 3431, 1076, 2789, 2235, 3209, 2158, 2298, 2393, 339, 1512, 490, 3135, 2350, 3471, 2469, 1404, 257, 4071, 2927, 94, 3994, 1218, 2863, 2944, 363, 2608, 462, 575, 1571, 1793, 1415, 580, 588, 3656, 3893, 3872, 2168, 2958, 788, 2622, 2459, 3999, 1272, 3852, 2050, 3491, 108, 2883, 474, 2036, 1092, 1524, 975, 1307, 2077, 310, 3946, 3979], ('v_proj', 16): [2533, 3431, 2789, 1076, 3209, 2235, 2158, 2298, 2393, 490, 3135, 3471, 1512, 1404, 257, 4071, 2350, 94, 2927, 339, 2469, 1218, 2863, 2944, 2608, 363, 3994, 575, 462, 1793, 1571, 580, 3656, 588, 3893, 1415, 3872, 788, 2958, 2168, 2622, 2459, 2050, 3852, 1272, 3491, 3999, 2883, 1524, 474, 3979, 108, 1307, 2077, 2036, 975, 375, 310, 3946, 1092], ('q_proj', 16): [2533, 3431, 2789, 1076, 3209, 2235, 2158, 2298, 2393, 490, 3135, 3471, 1512, 1404, 257, 4071, 2350, 94, 2927, 339, 2469, 1218, 2863, 2944, 2608, 363, 3994, 575, 462, 1793, 1571, 580, 3656, 588, 3893, 1415, 3872, 788, 2958, 2168, 2622, 2459, 2050, 3852, 1272, 3491, 3999, 2883, 1524, 474, 3979, 108, 1307, 2077, 2036, 975, 375, 310, 3946, 1092], ('k_proj', 16): [2533, 3431, 2789, 1076, 3209, 2235, 2158, 2298, 2393, 490, 3135, 3471, 1512, 1404, 257, 4071, 2350, 94, 2927, 339, 2469, 1218, 2863, 2944, 2608, 363, 3994, 575, 462, 1793, 1571, 580, 3656, 588, 3893, 1415, 3872, 788, 2958, 2168, 2622, 2459, 2050, 3852, 1272, 3491, 3999, 2883, 1524, 474, 3979, 108, 1307, 2077, 2036, 975, 375, 310, 3946, 1092], ('v_proj', 20): [2533, 3431, 1076, 2789, 3209, 2235, 2158, 2298, 2393, 1512, 3135, 3471, 4071, 1404, 490, 257, 2944, 3994, 2927, 363, 462, 575, 1218, 1793, 339, 2350, 94, 2863, 2608, 580, 2469, 3893, 1571, 2958, 3656, 2622, 788, 588, 2050, 3852, 1307, 2459, 3946, 3872, 2077, 1524, 375, 2168, 972, 310, 1163, 2358, 3979, 2573, 1548, 239, 2883, 3526, 2611, 1868, 2736, 3117, 3491, 309, 1489, 851, 886, 186, 3273, 1583, 2412, 299, 1930, 1167, 2271, 2742, 3862, 497, 1057], ('q_proj', 20): [2533, 3431, 1076, 2789, 3209, 2235, 2158, 2298, 2393, 1512, 3135, 3471, 4071, 1404, 490, 257, 2944, 3994, 2927, 363, 462, 575, 1218, 1793, 339, 2350, 94, 2863, 2608, 580, 2469, 3893, 1571, 2958, 3656, 2622, 788, 588, 2050, 3852, 1307, 2459, 3946, 3872, 2077, 1524, 375, 2168, 972, 310, 1163, 2358, 3979, 2573, 1548, 239, 2883, 3526, 2611, 1868, 2736, 3117, 3491, 309, 1489, 851, 886, 186, 3273, 1583, 2412, 299, 1930, 1167, 2271, 2742, 3862, 497, 1057], ('k_proj', 20): [2533, 3431, 1076, 2789, 3209, 2235, 2158, 2298, 2393, 1512, 3135, 3471, 4071, 1404, 490, 257, 2944, 3994, 2927, 363, 462, 575, 1218, 1793, 339, 2350, 94, 2863, 2608, 580, 2469, 3893, 1571, 2958, 3656, 2622, 788, 588, 2050, 3852, 1307, 2459, 3946, 3872, 2077, 1524, 375, 2168, 972, 310, 1163, 2358, 3979, 2573, 1548, 239, 2883, 3526, 2611, 1868, 2736, 3117, 3491, 309, 1489, 851, 886, 186, 3273, 1583, 2412, 299, 1930, 1167, 2271, 2742, 3862, 497, 1057], ('v_proj', 17): [2533, 3431, 2789, 1076, 3209, 2298, 2235, 2158, 2393, 3135, 1512, 3471, 490, 1404, 4071, 2350, 257, 2927, 339, 2469, 2944, 94, 2863, 1218, 462, 363, 3994, 575, 2608, 1571, 1793, 580, 3893, 3656, 588, 2958, 788, 2622, 3872, 2050, 2459, 3852, 2168, 1524, 3946, 1307, 2077, 375, 1272, 3979, 3491, 310, 2736, 972, 1163, 513, 3999, 2036, 2358, 239, 975, 883, 108], ('q_proj', 17): [2533, 3431, 2789, 1076, 3209, 2298, 2235, 2158, 2393, 3135, 1512, 3471, 490, 1404, 4071, 2350, 257, 2927, 339, 2469, 2944, 94, 2863, 1218, 462, 363, 3994, 575, 2608, 1571, 1793, 580, 3893, 3656, 588, 2958, 788, 2622, 3872, 2050, 2459, 3852, 2168, 1524, 3946, 1307, 2077, 375, 1272, 3979, 3491, 310, 2736, 972, 1163, 513, 3999, 2036, 2358, 239, 975, 883, 108], ('k_proj', 17): [2533, 3431, 2789, 1076, 3209, 2298, 2235, 2158, 2393, 3135, 1512, 3471, 490, 1404, 4071, 2350, 257, 2927, 339, 2469, 2944, 94, 2863, 1218, 462, 363, 3994, 575, 2608, 1571, 1793, 580, 3893, 3656, 588, 2958, 788, 2622, 3872, 2050, 2459, 3852, 2168, 1524, 3946, 1307, 2077, 375, 1272, 3979, 3491, 310, 2736, 972, 1163, 513, 3999, 2036, 2358, 239, 975, 883, 108], ('v_proj', 13): [2533, 1076, 3431, 2789, 2158, 2393, 3209, 2350, 339, 2235, 257, 2298, 490, 2469, 94, 1512, 4071, 1404, 2608, 2927, 1218, 3994, 462, 1571, 363, 3135, 2863, 1793, 3999, 2168, 588, 3656, 3471, 2459, 1272, 2944, 788, 3872, 3893, 2622, 2050, 1583, 474, 575, 2036, 3491, 2958, 580], ('q_proj', 13): [2533, 1076, 3431, 2789, 2158, 2393, 3209, 2350, 339, 2235, 257, 2298, 490, 2469, 94, 1512, 4071, 1404, 2608, 2927, 1218, 3994, 462, 1571, 363, 3135, 2863, 1793, 3999, 2168, 588, 3656, 3471, 2459, 1272, 2944, 788, 3872, 3893, 2622, 2050, 1583, 474, 575, 2036, 3491, 2958, 580], ('k_proj', 13): [2533, 1076, 3431, 2789, 2158, 2393, 3209, 2350, 339, 2235, 257, 2298, 490, 2469, 94, 1512, 4071, 1404, 2608, 2927, 1218, 3994, 462, 1571, 363, 3135, 2863, 1793, 3999, 2168, 588, 3656, 3471, 2459, 1272, 2944, 788, 3872, 3893, 2622, 2050, 1583, 474, 575, 2036, 3491, 2958, 580], ('v_proj', 18): [2533, 3431, 1076, 2789, 3209, 2235, 2298, 2158, 1512, 2393, 3135, 3471, 490, 4071, 1404, 2927, 2944, 257, 94, 339, 3994, 575, 2863, 2350, 363, 1218, 462, 2469, 2608, 580, 1571, 3893, 1793, 2958, 3656, 588, 788, 2622, 3872, 2050, 3852, 2168, 2459, 3946, 1307, 310, 1524, 375, 2077, 972, 3979, 1272, 1163, 1868, 2358, 2736, 513, 3819, 2389, 130, 3273, 3491, 3117, 1680, 2573, 975, 239, 2036, 2883, 340, 2195, 883, 497, 2412, 1548, 886, 2742], ('q_proj', 18): [2533, 3431, 1076, 2789, 3209, 2235, 2298, 2158, 1512, 2393, 3135, 3471, 490, 4071, 1404, 2927, 2944, 257, 94, 339, 3994, 575, 2863, 2350, 363, 1218, 462, 2469, 2608, 580, 1571, 3893, 1793, 2958, 3656, 588, 788, 2622, 3872, 2050, 3852, 2168, 2459, 3946, 1307, 310, 1524, 375, 2077, 972, 3979, 1272, 1163, 1868, 2358, 2736, 513, 3819, 2389, 130, 3273, 3491, 3117, 1680, 2573, 975, 239, 2036, 2883, 340, 2195, 883, 497, 2412, 1548, 886, 2742], ('k_proj', 18): [2533, 3431, 1076, 2789, 3209, 2235, 2298, 2158, 1512, 2393, 3135, 3471, 490, 4071, 1404, 2927, 2944, 257, 94, 339, 3994, 575, 2863, 2350, 363, 1218, 462, 2469, 2608, 580, 1571, 3893, 1793, 2958, 3656, 588, 788, 2622, 3872, 2050, 3852, 2168, 2459, 3946, 1307, 310, 1524, 375, 2077, 972, 3979, 1272, 1163, 1868, 2358, 2736, 513, 3819, 2389, 130, 3273, 3491, 3117, 1680, 2573, 975, 239, 2036, 2883, 340, 2195, 883, 497, 2412, 1548, 886, 2742], ('v_proj', 24): [2533, 2789, 3431, 1076, 2158, 3209, 2298, 2235, 2393, 3135, 1512, 4071, 363, 1404, 3471, 257, 2944, 575, 3994, 462, 2608, 580, 490, 2469, 1793, 339, 3893, 94, 1218, 2350, 2883, 2958, 3656, 2863, 2927, 2622, 788, 1571, 2050, 588, 3946, 3852, 2459, 1307, 2077, 375, 310, 1524, 239, 1163, 2281, 2168, 2358, 2611, 972, 1057, 3117, 1523, 1721, 1868, 886, 3536, 3391, 4063, 2016, 3491, 1825, 3148, 1167, 3921, 3526, 3249, 866, 1316, 1548, 1489, 707, 2573, 200, 1583, 3514, 1378, 134, 3562, 2736, 685, 1734, 1449, 1117, 299, 487, 1119, 2248, 2187, 1930, 3629, 3840, 739, 2140, 1409, 3872, 3106, 2226, 3772, 2825, 2801, 309, 2403, 1360, 2199, 3674, 3290, 1399, 2147, 2522, 1877, 2540, 3836, 1888, 2949, 2377, 3722, 2433, 368, 2581, 1374, 1619, 2204, 3999, 1492, 956, 1053, 345, 1174, 3205, 2718, 2476, 1134, 2869, 1545, 3816, 2688, 1269, 4085, 3270, 1151, 3556, 2687, 448, 3490, 3731, 1062, 3333, 1510, 3767, 3819, 3807, 3245, 470, 3579, 1113, 4076, 1772, 745, 1354, 3051, 378, 2131, 2192, 856, 3362, 3065, 4048, 1461, 2588, 1090, 150, 1272, 66, 2018, 2815, 2740], ('q_proj', 24): [2533, 2789, 3431, 1076, 2158, 3209, 2298, 2235, 2393, 3135, 1512, 4071, 363, 1404, 3471, 257, 2944, 575, 3994, 462, 2608, 580, 490, 2469, 1793, 339, 3893, 94, 1218, 2350, 2883, 2958, 3656, 2863, 2927, 2622, 788, 1571, 2050, 588, 3946, 3852, 2459, 1307, 2077, 375, 310, 1524, 239, 1163, 2281, 2168, 2358, 2611, 972, 1057, 3117, 1523, 1721, 1868, 886, 3536, 3391, 4063, 2016, 3491, 1825, 3148, 1167, 3921, 3526, 3249, 866, 1316, 1548, 1489, 707, 2573, 200, 1583, 3514, 1378, 134, 3562, 2736, 685, 1734, 1449, 1117, 299, 487, 1119, 2248, 2187, 1930, 3629, 3840, 739, 2140, 1409, 3872, 3106, 2226, 3772, 2825, 2801, 309, 2403, 1360, 2199, 3674, 3290, 1399, 2147, 2522, 1877, 2540, 3836, 1888, 2949, 2377, 3722, 2433, 368, 2581, 1374, 1619, 2204, 3999, 1492, 956, 1053, 345, 1174, 3205, 2718, 2476, 1134, 2869, 1545, 3816, 2688, 1269, 4085, 3270, 1151, 3556, 2687, 448, 3490, 3731, 1062, 3333, 1510, 3767, 3819, 3807, 3245, 470, 3579, 1113, 4076, 1772, 745, 1354, 3051, 378, 2131, 2192, 856, 3362, 3065, 4048, 1461, 2588, 1090, 150, 1272, 66, 2018, 2815, 2740], ('k_proj', 24): [2533, 2789, 3431, 1076, 2158, 3209, 2298, 2235, 2393, 3135, 1512, 4071, 363, 1404, 3471, 257, 2944, 575, 3994, 462, 2608, 580, 490, 2469, 1793, 339, 3893, 94, 1218, 2350, 2883, 2958, 3656, 2863, 2927, 2622, 788, 1571, 2050, 588, 3946, 3852, 2459, 1307, 2077, 375, 310, 1524, 239, 1163, 2281, 2168, 2358, 2611, 972, 1057, 3117, 1523, 1721, 1868, 886, 3536, 3391, 4063, 2016, 3491, 1825, 3148, 1167, 3921, 3526, 3249, 866, 1316, 1548, 1489, 707, 2573, 200, 1583, 3514, 1378, 134, 3562, 2736, 685, 1734, 1449, 1117, 299, 487, 1119, 2248, 2187, 1930, 3629, 3840, 739, 2140, 1409, 3872, 3106, 2226, 3772, 2825, 2801, 309, 2403, 1360, 2199, 3674, 3290, 1399, 2147, 2522, 1877, 2540, 3836, 1888, 2949, 2377, 3722, 2433, 368, 2581, 1374, 1619, 2204, 3999, 1492, 956, 1053, 345, 1174, 3205, 2718, 2476, 1134, 2869, 1545, 3816, 2688, 1269, 4085, 3270, 1151, 3556, 2687, 448, 3490, 3731, 1062, 3333, 1510, 3767, 3819, 3807, 3245, 470, 3579, 1113, 4076, 1772, 745, 1354, 3051, 378, 2131, 2192, 856, 3362, 3065, 4048, 1461, 2588, 1090, 150, 1272, 66, 2018, 2815, 2740], ('v_proj', 28): [2533, 3431, 2789, 2298, 1076, 2158, 3209, 1512, 3135, 2235, 2393, 4071, 2944, 2883, 575, 1404, 257, 3994, 580, 363, 3893, 339, 462, 2469, 1793, 3471, 2350, 2622, 2958, 1218, 3656, 2608, 2927, 490, 1571, 2077, 2050, 1307, 2281, 788, 3946, 3852, 2459, 375, 3391, 94, 239, 588, 1360, 1316, 1734, 4063, 1524, 2358, 1825, 1167, 2611, 470, 568, 1721, 1134, 4053, 3536, 2639, 2095, 803, 3952, 2016, 866, 3556, 3117, 2863, 2654, 3148, 3771, 2476, 1117, 2018, 3807, 3270, 2403, 1057, 2543, 2279, 2248, 2627, 556, 3772, 2204, 299, 2750, 3839, 1548, 2644, 707, 1461, 1523, 886, 3065, 2168, 3362, 1374, 2924, 2026, 972, 1583, 2573, 1163, 4082, 669, 3819, 149, 3921, 1868, 1501, 3700, 1492, 2365, 3169, 3830, 1551, 1545, 3335, 3386, 985, 2625, 358, 1510, 310, 1174, 988, 666, 1062, 2131, 1261, 1926, 1755, 1017, 2226, 1944, 3722, 1863, 188, 2852, 252, 102, 3290, 1409, 3205, 2551, 3836, 2140, 805, 500, 2633, 587, 3571, 1051, 1930, 2869, 739, 685, 1888, 795, 3816, 3512, 2815, 3249, 824, 1272, 3067, 2022, 3629, 662, 1844, 1637, 3, 446, 3854, 3710, 2825, 2540, 729, 2282, 3324, 3168, 3674, 368, 2036, 1335, 3445, 2561, 2612, 45, 3579, 911, 745, 3075, 448, 747, 1296, 2038, 2579, 1857, 2718, 3731, 3840, 1392, 505, 2326, 2688, 134, 3577, 3999, 1146, 1334, 3619, 2410, 3449, 2203, 1841, 2881, 2205, 2048, 367, 2180, 3178, 1054, 2578, 2333, 2880, 3310, 200, 3123, 309, 2801, 4085, 2254, 1688, 790, 1626, 2070, 1119, 3106, 2340, 3814, 3767, 4055, 3245, 2968, 3506, 3334, 1130, 2760, 1664, 1354, 3444, 4076, 1399, 2199, 814, 189, 1473, 1025, 3057, 3236, 968, 2411, 1915, 1763, 1819, 3064, 2295, 1635, 3803, 1007, 3562, 207, 1580, 3390, 4077, 1090, 156, 1569, 237, 1746, 176, 1686, 204, 1972, 2397, 1468, 1748, 2949, 712, 304, 1947, 1861, 3151, 292, 1768, 70, 2581, 1425, 1045, 3911, 1656, 596, 1489, 213, 654, 1449, 178, 1481, 2780, 328, 1257, 2934, 3241, 2913, 1752, 3498, 809, 3837, 3172, 221, 2150, 3174, 555, 1183, 3319, 1998, 513, 2104, 786, 1088, 2272, 2275, 3072, 3865, 3435, 3402, 2975, 1041, 378, 3529, 1511, 35, 2043, 216, 222, 3427, 1103, 1091, 2772, 2658, 3653, 1574, 2959, 1068, 2803, 3173, 3768, 2763, 1529, 2413, 413, 3465, 3045, 4070, 2839, 759, 3567, 2433, 3778, 1113, 4046, 7, 1718, 655, 3194, 1761, 3051, 606, 1075, 3268, 1679, 3662, 589, 388, 955, 3825, 443, 1330, 2381, 1285, 1377, 1636, 1877, 1472, 2420, 2290, 831, 3640, 544, 2998, 1073, 949, 1127, 4018, 3981, 1629, 1171, 2805, 3292, 3289, 956, 2513, 2375, 4081, 1661, 1344, 352, 725, 306, 1648, 3539, 473, 4090, 3995, 2217, 2232, 2061, 1244, 3430, 3558, 1100, 3092, 1592, 999, 2177, 1242, 3080, 28, 3348, 351, 3689, 116, 1802, 245, 856, 3280, 2377, 148, 2085, 2310, 397, 2709, 3966, 1180, 1269, 2915, 619, 3442, 3953, 372, 3866, 2458, 2237, 3333, 3582, 3977, 2867, 2447, 3842, 1966, 3783, 2952, 1058, 3818, 1986, 3246, 2523, 2520, 3328, 883, 2439, 3251, 2904, 317, 3976, 1619, 1897, 66, 2565, 2160, 3130, 2643, 1549, 819, 1289, 3835, 2405, 155, 2187, 1228, 2114, 2443, 2354, 3001, 3247, 3397, 2030, 1094, 1625, 2607, 2768, 174], ('q_proj', 28): [2533, 3431, 2789, 2298, 1076, 2158, 3209, 1512, 3135, 2235, 2393, 4071, 2944, 2883, 575, 1404, 257, 3994, 580, 363, 3893, 339, 462, 2469, 1793, 3471, 2350, 2622, 2958, 1218, 3656, 2608, 2927, 490, 1571, 2077, 2050, 1307, 2281, 788, 3946, 3852, 2459, 375, 3391, 94, 239, 588, 1360, 1316, 1734, 4063, 1524, 2358, 1825, 1167, 2611, 470, 568, 1721, 1134, 4053, 3536, 2639, 2095, 803, 3952, 2016, 866, 3556, 3117, 2863, 2654, 3148, 3771, 2476, 1117, 2018, 3807, 3270, 2403, 1057, 2543, 2279, 2248, 2627, 556, 3772, 2204, 299, 2750, 3839, 1548, 2644, 707, 1461, 1523, 886, 3065, 2168, 3362, 1374, 2924, 2026, 972, 1583, 2573, 1163, 4082, 669, 3819, 149, 3921, 1868, 1501, 3700, 1492, 2365, 3169, 3830, 1551, 1545, 3335, 3386, 985, 2625, 358, 1510, 310, 1174, 988, 666, 1062, 2131, 1261, 1926, 1755, 1017, 2226, 1944, 3722, 1863, 188, 2852, 252, 102, 3290, 1409, 3205, 2551, 3836, 2140, 805, 500, 2633, 587, 3571, 1051, 1930, 2869, 739, 685, 1888, 795, 3816, 3512, 2815, 3249, 824, 1272, 3067, 2022, 3629, 662, 1844, 1637, 3, 446, 3854, 3710, 2825, 2540, 729, 2282, 3324, 3168, 3674, 368, 2036, 1335, 3445, 2561, 2612, 45, 3579, 911, 745, 3075, 448, 747, 1296, 2038, 2579, 1857, 2718, 3731, 3840, 1392, 505, 2326, 2688, 134, 3577, 3999, 1146, 1334, 3619, 2410, 3449, 2203, 1841, 2881, 2205, 2048, 367, 2180, 3178, 1054, 2578, 2333, 2880, 3310, 200, 3123, 309, 2801, 4085, 2254, 1688, 790, 1626, 2070, 1119, 3106, 2340, 3814, 3767, 4055, 3245, 2968, 3506, 3334, 1130, 2760, 1664, 1354, 3444, 4076, 1399, 2199, 814, 189, 1473, 1025, 3057, 3236, 968, 2411, 1915, 1763, 1819, 3064, 2295, 1635, 3803, 1007, 3562, 207, 1580, 3390, 4077, 1090, 156, 1569, 237, 1746, 176, 1686, 204, 1972, 2397, 1468, 1748, 2949, 712, 304, 1947, 1861, 3151, 292, 1768, 70, 2581, 1425, 1045, 3911, 1656, 596, 1489, 213, 654, 1449, 178, 1481, 2780, 328, 1257, 2934, 3241, 2913, 1752, 3498, 809, 3837, 3172, 221, 2150, 3174, 555, 1183, 3319, 1998, 513, 2104, 786, 1088, 2272, 2275, 3072, 3865, 3435, 3402, 2975, 1041, 378, 3529, 1511, 35, 2043, 216, 222, 3427, 1103, 1091, 2772, 2658, 3653, 1574, 2959, 1068, 2803, 3173, 3768, 2763, 1529, 2413, 413, 3465, 3045, 4070, 2839, 759, 3567, 2433, 3778, 1113, 4046, 7, 1718, 655, 3194, 1761, 3051, 606, 1075, 3268, 1679, 3662, 589, 388, 955, 3825, 443, 1330, 2381, 1285, 1377, 1636, 1877, 1472, 2420, 2290, 831, 3640, 544, 2998, 1073, 949, 1127, 4018, 3981, 1629, 1171, 2805, 3292, 3289, 956, 2513, 2375, 4081, 1661, 1344, 352, 725, 306, 1648, 3539, 473, 4090, 3995, 2217, 2232, 2061, 1244, 3430, 3558, 1100, 3092, 1592, 999, 2177, 1242, 3080, 28, 3348, 351, 3689, 116, 1802, 245, 856, 3280, 2377, 148, 2085, 2310, 397, 2709, 3966, 1180, 1269, 2915, 619, 3442, 3953, 372, 3866, 2458, 2237, 3333, 3582, 3977, 2867, 2447, 3842, 1966, 3783, 2952, 1058, 3818, 1986, 3246, 2523, 2520, 3328, 883, 2439, 3251, 2904, 317, 3976, 1619, 1897, 66, 2565, 2160, 3130, 2643, 1549, 819, 1289, 3835, 2405, 155, 2187, 1228, 2114, 2443, 2354, 3001, 3247, 3397, 2030, 1094, 1625, 2607, 2768, 174], ('k_proj', 28): [2533, 3431, 2789, 2298, 1076, 2158, 3209, 1512, 3135, 2235, 2393, 4071, 2944, 2883, 575, 1404, 257, 3994, 580, 363, 3893, 339, 462, 2469, 1793, 3471, 2350, 2622, 2958, 1218, 3656, 2608, 2927, 490, 1571, 2077, 2050, 1307, 2281, 788, 3946, 3852, 2459, 375, 3391, 94, 239, 588, 1360, 1316, 1734, 4063, 1524, 2358, 1825, 1167, 2611, 470, 568, 1721, 1134, 4053, 3536, 2639, 2095, 803, 3952, 2016, 866, 3556, 3117, 2863, 2654, 3148, 3771, 2476, 1117, 2018, 3807, 3270, 2403, 1057, 2543, 2279, 2248, 2627, 556, 3772, 2204, 299, 2750, 3839, 1548, 2644, 707, 1461, 1523, 886, 3065, 2168, 3362, 1374, 2924, 2026, 972, 1583, 2573, 1163, 4082, 669, 3819, 149, 3921, 1868, 1501, 3700, 1492, 2365, 3169, 3830, 1551, 1545, 3335, 3386, 985, 2625, 358, 1510, 310, 1174, 988, 666, 1062, 2131, 1261, 1926, 1755, 1017, 2226, 1944, 3722, 1863, 188, 2852, 252, 102, 3290, 1409, 3205, 2551, 3836, 2140, 805, 500, 2633, 587, 3571, 1051, 1930, 2869, 739, 685, 1888, 795, 3816, 3512, 2815, 3249, 824, 1272, 3067, 2022, 3629, 662, 1844, 1637, 3, 446, 3854, 3710, 2825, 2540, 729, 2282, 3324, 3168, 3674, 368, 2036, 1335, 3445, 2561, 2612, 45, 3579, 911, 745, 3075, 448, 747, 1296, 2038, 2579, 1857, 2718, 3731, 3840, 1392, 505, 2326, 2688, 134, 3577, 3999, 1146, 1334, 3619, 2410, 3449, 2203, 1841, 2881, 2205, 2048, 367, 2180, 3178, 1054, 2578, 2333, 2880, 3310, 200, 3123, 309, 2801, 4085, 2254, 1688, 790, 1626, 2070, 1119, 3106, 2340, 3814, 3767, 4055, 3245, 2968, 3506, 3334, 1130, 2760, 1664, 1354, 3444, 4076, 1399, 2199, 814, 189, 1473, 1025, 3057, 3236, 968, 2411, 1915, 1763, 1819, 3064, 2295, 1635, 3803, 1007, 3562, 207, 1580, 3390, 4077, 1090, 156, 1569, 237, 1746, 176, 1686, 204, 1972, 2397, 1468, 1748, 2949, 712, 304, 1947, 1861, 3151, 292, 1768, 70, 2581, 1425, 1045, 3911, 1656, 596, 1489, 213, 654, 1449, 178, 1481, 2780, 328, 1257, 2934, 3241, 2913, 1752, 3498, 809, 3837, 3172, 221, 2150, 3174, 555, 1183, 3319, 1998, 513, 2104, 786, 1088, 2272, 2275, 3072, 3865, 3435, 3402, 2975, 1041, 378, 3529, 1511, 35, 2043, 216, 222, 3427, 1103, 1091, 2772, 2658, 3653, 1574, 2959, 1068, 2803, 3173, 3768, 2763, 1529, 2413, 413, 3465, 3045, 4070, 2839, 759, 3567, 2433, 3778, 1113, 4046, 7, 1718, 655, 3194, 1761, 3051, 606, 1075, 3268, 1679, 3662, 589, 388, 955, 3825, 443, 1330, 2381, 1285, 1377, 1636, 1877, 1472, 2420, 2290, 831, 3640, 544, 2998, 1073, 949, 1127, 4018, 3981, 1629, 1171, 2805, 3292, 3289, 956, 2513, 2375, 4081, 1661, 1344, 352, 725, 306, 1648, 3539, 473, 4090, 3995, 2217, 2232, 2061, 1244, 3430, 3558, 1100, 3092, 1592, 999, 2177, 1242, 3080, 28, 3348, 351, 3689, 116, 1802, 245, 856, 3280, 2377, 148, 2085, 2310, 397, 2709, 3966, 1180, 1269, 2915, 619, 3442, 3953, 372, 3866, 2458, 2237, 3333, 3582, 3977, 2867, 2447, 3842, 1966, 3783, 2952, 1058, 3818, 1986, 3246, 2523, 2520, 3328, 883, 2439, 3251, 2904, 317, 3976, 1619, 1897, 66, 2565, 2160, 3130, 2643, 1549, 819, 1289, 3835, 2405, 155, 2187, 1228, 2114, 2443, 2354, 3001, 3247, 3397, 2030, 1094, 1625, 2607, 2768, 174], ('v_proj', 22): [2533, 3431, 2789, 1076, 3209, 2235, 1512, 2298, 2158, 2393, 3135, 3471, 4071, 1404, 257, 363, 2944, 575, 490, 2927, 3994, 94, 462, 1793, 580, 2608, 1218, 3893, 2350, 2863, 2469, 339, 2958, 2622, 3656, 1571, 788, 588, 2050, 3852, 2883, 3946, 2459, 1307, 2077, 375, 1524, 1868, 2358, 3526, 972, 310, 1163, 3872, 2611, 239, 2168, 3921, 2281, 1489, 1583, 3117, 3536, 1057, 1316, 2736, 1523, 3979, 3491, 2573, 345, 1167, 299, 1930, 1548, 866, 1119, 3514, 886, 487, 2016, 309, 200, 4063, 1721, 130, 3562, 134, 707, 186, 685, 2248, 1825, 3273, 2199, 1374, 2581, 3840, 2204, 3674, 1378, 1053, 2140, 2522, 2801, 1272, 340, 1551, 3031, 2750, 2988, 2147, 851, 1619, 3807, 1510, 2412, 2949, 2389, 1134, 3722, 3816, 968, 3836, 2959, 883, 3862, 3556, 2688], ('q_proj', 22): [2533, 3431, 2789, 1076, 3209, 2235, 1512, 2298, 2158, 2393, 3135, 3471, 4071, 1404, 257, 363, 2944, 575, 490, 2927, 3994, 94, 462, 1793, 580, 2608, 1218, 3893, 2350, 2863, 2469, 339, 2958, 2622, 3656, 1571, 788, 588, 2050, 3852, 2883, 3946, 2459, 1307, 2077, 375, 1524, 1868, 2358, 3526, 972, 310, 1163, 3872, 2611, 239, 2168, 3921, 2281, 1489, 1583, 3117, 3536, 1057, 1316, 2736, 1523, 3979, 3491, 2573, 345, 1167, 299, 1930, 1548, 866, 1119, 3514, 886, 487, 2016, 309, 200, 4063, 1721, 130, 3562, 134, 707, 186, 685, 2248, 1825, 3273, 2199, 1374, 2581, 3840, 2204, 3674, 1378, 1053, 2140, 2522, 2801, 1272, 340, 1551, 3031, 2750, 2988, 2147, 851, 1619, 3807, 1510, 2412, 2949, 2389, 1134, 3722, 3816, 968, 3836, 2959, 883, 3862, 3556, 2688], ('k_proj', 22): [2533, 3431, 2789, 1076, 3209, 2235, 1512, 2298, 2158, 2393, 3135, 3471, 4071, 1404, 257, 363, 2944, 575, 490, 2927, 3994, 94, 462, 1793, 580, 2608, 1218, 3893, 2350, 2863, 2469, 339, 2958, 2622, 3656, 1571, 788, 588, 2050, 3852, 2883, 3946, 2459, 1307, 2077, 375, 1524, 1868, 2358, 3526, 972, 310, 1163, 3872, 2611, 239, 2168, 3921, 2281, 1489, 1583, 3117, 3536, 1057, 1316, 2736, 1523, 3979, 3491, 2573, 345, 1167, 299, 1930, 1548, 866, 1119, 3514, 886, 487, 2016, 309, 200, 4063, 1721, 130, 3562, 134, 707, 186, 685, 2248, 1825, 3273, 2199, 1374, 2581, 3840, 2204, 3674, 1378, 1053, 2140, 2522, 2801, 1272, 340, 1551, 3031, 2750, 2988, 2147, 851, 1619, 3807, 1510, 2412, 2949, 2389, 1134, 3722, 3816, 968, 3836, 2959, 883, 3862, 3556, 2688], ('v_proj', 19): [2533, 3431, 1076, 2789, 2235, 3209, 2298, 2158, 2393, 1512, 3135, 3471, 4071, 1404, 490, 2927, 257, 2944, 94, 363, 462, 575, 2863, 3994, 1218, 2608, 1793, 2350, 2469, 339, 580, 3893, 1571, 3656, 2958, 588, 2622, 788, 2050, 3852, 3872, 3946, 2168, 2459, 1307, 1524, 2077, 375, 1163, 972, 3979, 310, 239, 513, 3491, 886, 1548, 1868, 2358, 3117, 2883, 130, 2573, 1272, 497, 851, 186, 3273, 3862, 749, 2412, 1510, 975, 1489, 299, 2736, 883, 2036], ('q_proj', 19): [2533, 3431, 1076, 2789, 2235, 3209, 2298, 2158, 2393, 1512, 3135, 3471, 4071, 1404, 490, 2927, 257, 2944, 94, 363, 462, 575, 2863, 3994, 1218, 2608, 1793, 2350, 2469, 339, 580, 3893, 1571, 3656, 2958, 588, 2622, 788, 2050, 3852, 3872, 3946, 2168, 2459, 1307, 1524, 2077, 375, 1163, 972, 3979, 310, 239, 513, 3491, 886, 1548, 1868, 2358, 3117, 2883, 130, 2573, 1272, 497, 851, 186, 3273, 3862, 749, 2412, 1510, 975, 1489, 299, 2736, 883, 2036], ('k_proj', 19): [2533, 3431, 1076, 2789, 2235, 3209, 2298, 2158, 2393, 1512, 3135, 3471, 4071, 1404, 490, 2927, 257, 2944, 94, 363, 462, 575, 2863, 3994, 1218, 2608, 1793, 2350, 2469, 339, 580, 3893, 1571, 3656, 2958, 588, 2622, 788, 2050, 3852, 3872, 3946, 2168, 2459, 1307, 1524, 2077, 375, 1163, 972, 3979, 310, 239, 513, 3491, 886, 1548, 1868, 2358, 3117, 2883, 130, 2573, 1272, 497, 851, 186, 3273, 3862, 749, 2412, 1510, 975, 1489, 299, 2736, 883, 2036], ('v_proj', 27): [2533, 3431, 1512, 2789, 1076, 3209, 2298, 2158, 2235, 3135, 2393, 2944, 4071, 2883, 363, 575, 257, 1404, 3994, 580, 3893, 1793, 462, 3471, 339, 2469, 2958, 1218, 2608, 2350, 2622, 3656, 1571, 2077, 2050, 490, 788, 2927, 1307, 94, 3946, 3852, 375, 2459, 588, 2281, 239, 1524, 3536, 2863, 1117, 2016, 1316, 2358, 4063, 866, 1825, 1734, 886, 1721, 3148, 972, 3556, 2403, 2625, 3117, 2611, 1057, 3771, 2476, 1548, 2168, 299, 2627, 1167, 3391, 1868, 3819, 2248, 1523, 2018, 2573, 1510, 707, 1360, 1501, 803, 3722, 3065, 1888, 685, 2095, 1163, 3386, 1583, 513, 3807, 2654, 3952, 3270, 470, 2750, 2365, 2543, 358, 3772, 3249, 1130, 556, 3579, 3854, 1134, 1399, 1944, 1374, 3710, 1062, 2279, 3767, 252, 1017, 1551, 310, 1272, 3921, 3106, 1492, 2204, 2924, 1461, 2205, 3245, 1119, 3335, 2581, 1930, 3816, 3362, 3, 666, 2226, 1926, 1545, 3080, 3445, 2377, 669, 2540, 2131, 2869, 1409, 3731, 1449, 911, 1257, 2282, 955, 3512, 1755, 739, 3836, 2815, 149, 4082, 3629, 2026, 2326, 1857, 2410, 1180, 2688, 4053, 134, 2551, 200, 2140, 3529, 2199, 2644, 568, 2272, 759, 587, 3803, 448, 3205, 367, 2968, 3562, 3839, 1635, 3333, 1051, 745, 3075, 2612, 188, 2038, 1664, 1146, 3619, 2070, 45, 368, 3067, 1354, 2825, 3999, 809, 729, 500, 371, 3290, 102, 3194, 985, 814, 3172, 2718, 2780, 446, 2633, 1473, 2881, 2048, 2949, 3253, 1661, 1183, 3051, 1468, 662, 3674, 805, 1656, 3814, 795, 3427, 3840, 189, 3151, 378, 1637, 3866, 747, 1091, 1261, 3662, 1334, 1863, 3700, 3577, 750, 3072, 1819, 712, 3571, 3024, 1392, 1620, 2801, 3324, 988, 3830, 1174, 3257, 1626, 3539, 1636, 3567, 1686, 1688, 2043, 2022, 1972, 544, 2036, 3981, 2578, 857, 884, 555, 3526, 158, 66, 2768, 1648, 1269, 3168, 2340, 3506, 3863, 328, 213, 3514, 1861, 3241, 304, 1090, 2579, 2254, 3247, 2520, 178, 2896, 1115, 1592, 1841, 2290, 1041, 3449, 455, 2998, 1045, 2180, 511, 3268, 2447, 3310, 1335, 309, 6, 1007, 3057, 1489, 2736, 883, 2759, 824, 3966, 3045, 2928, 2275, 1619, 856, 221, 443, 3818, 1296, 3334, 406, 2192, 827, 3246, 174, 4085, 3173, 1752, 4055, 2572, 2333, 2846, 1877, 4076, 3825, 2952, 2643, 505, 1073, 596, 4047, 413, 3251, 967, 156, 2993, 466, 237, 956, 968, 460, 2400, 2639, 2658, 70, 3348, 3586, 2187, 2240, 1171, 3289, 1046, 2232, 1748, 1481, 3092, 1856, 3842, 1113, 377, 2114, 2760, 2893, 487, 2295, 1176, 2772, 1998, 21, 3939, 2765], ('q_proj', 27): [2533, 3431, 1512, 2789, 1076, 3209, 2298, 2158, 2235, 3135, 2393, 2944, 4071, 2883, 363, 575, 257, 1404, 3994, 580, 3893, 1793, 462, 3471, 339, 2469, 2958, 1218, 2608, 2350, 2622, 3656, 1571, 2077, 2050, 490, 788, 2927, 1307, 94, 3946, 3852, 375, 2459, 588, 2281, 239, 1524, 3536, 2863, 1117, 2016, 1316, 2358, 4063, 866, 1825, 1734, 886, 1721, 3148, 972, 3556, 2403, 2625, 3117, 2611, 1057, 3771, 2476, 1548, 2168, 299, 2627, 1167, 3391, 1868, 3819, 2248, 1523, 2018, 2573, 1510, 707, 1360, 1501, 803, 3722, 3065, 1888, 685, 2095, 1163, 3386, 1583, 513, 3807, 2654, 3952, 3270, 470, 2750, 2365, 2543, 358, 3772, 3249, 1130, 556, 3579, 3854, 1134, 1399, 1944, 1374, 3710, 1062, 2279, 3767, 252, 1017, 1551, 310, 1272, 3921, 3106, 1492, 2204, 2924, 1461, 2205, 3245, 1119, 3335, 2581, 1930, 3816, 3362, 3, 666, 2226, 1926, 1545, 3080, 3445, 2377, 669, 2540, 2131, 2869, 1409, 3731, 1449, 911, 1257, 2282, 955, 3512, 1755, 739, 3836, 2815, 149, 4082, 3629, 2026, 2326, 1857, 2410, 1180, 2688, 4053, 134, 2551, 200, 2140, 3529, 2199, 2644, 568, 2272, 759, 587, 3803, 448, 3205, 367, 2968, 3562, 3839, 1635, 3333, 1051, 745, 3075, 2612, 188, 2038, 1664, 1146, 3619, 2070, 45, 368, 3067, 1354, 2825, 3999, 809, 729, 500, 371, 3290, 102, 3194, 985, 814, 3172, 2718, 2780, 446, 2633, 1473, 2881, 2048, 2949, 3253, 1661, 1183, 3051, 1468, 662, 3674, 805, 1656, 3814, 795, 3427, 3840, 189, 3151, 378, 1637, 3866, 747, 1091, 1261, 3662, 1334, 1863, 3700, 3577, 750, 3072, 1819, 712, 3571, 3024, 1392, 1620, 2801, 3324, 988, 3830, 1174, 3257, 1626, 3539, 1636, 3567, 1686, 1688, 2043, 2022, 1972, 544, 2036, 3981, 2578, 857, 884, 555, 3526, 158, 66, 2768, 1648, 1269, 3168, 2340, 3506, 3863, 328, 213, 3514, 1861, 3241, 304, 1090, 2579, 2254, 3247, 2520, 178, 2896, 1115, 1592, 1841, 2290, 1041, 3449, 455, 2998, 1045, 2180, 511, 3268, 2447, 3310, 1335, 309, 6, 1007, 3057, 1489, 2736, 883, 2759, 824, 3966, 3045, 2928, 2275, 1619, 856, 221, 443, 3818, 1296, 3334, 406, 2192, 827, 3246, 174, 4085, 3173, 1752, 4055, 2572, 2333, 2846, 1877, 4076, 3825, 2952, 2643, 505, 1073, 596, 4047, 413, 3251, 967, 156, 2993, 466, 237, 956, 968, 460, 2400, 2639, 2658, 70, 3348, 3586, 2187, 2240, 1171, 3289, 1046, 2232, 1748, 1481, 3092, 1856, 3842, 1113, 377, 2114, 2760, 2893, 487, 2295, 1176, 2772, 1998, 21, 3939, 2765], ('k_proj', 27): [2533, 3431, 1512, 2789, 1076, 3209, 2298, 2158, 2235, 3135, 2393, 2944, 4071, 2883, 363, 575, 257, 1404, 3994, 580, 3893, 1793, 462, 3471, 339, 2469, 2958, 1218, 2608, 2350, 2622, 3656, 1571, 2077, 2050, 490, 788, 2927, 1307, 94, 3946, 3852, 375, 2459, 588, 2281, 239, 1524, 3536, 2863, 1117, 2016, 1316, 2358, 4063, 866, 1825, 1734, 886, 1721, 3148, 972, 3556, 2403, 2625, 3117, 2611, 1057, 3771, 2476, 1548, 2168, 299, 2627, 1167, 3391, 1868, 3819, 2248, 1523, 2018, 2573, 1510, 707, 1360, 1501, 803, 3722, 3065, 1888, 685, 2095, 1163, 3386, 1583, 513, 3807, 2654, 3952, 3270, 470, 2750, 2365, 2543, 358, 3772, 3249, 1130, 556, 3579, 3854, 1134, 1399, 1944, 1374, 3710, 1062, 2279, 3767, 252, 1017, 1551, 310, 1272, 3921, 3106, 1492, 2204, 2924, 1461, 2205, 3245, 1119, 3335, 2581, 1930, 3816, 3362, 3, 666, 2226, 1926, 1545, 3080, 3445, 2377, 669, 2540, 2131, 2869, 1409, 3731, 1449, 911, 1257, 2282, 955, 3512, 1755, 739, 3836, 2815, 149, 4082, 3629, 2026, 2326, 1857, 2410, 1180, 2688, 4053, 134, 2551, 200, 2140, 3529, 2199, 2644, 568, 2272, 759, 587, 3803, 448, 3205, 367, 2968, 3562, 3839, 1635, 3333, 1051, 745, 3075, 2612, 188, 2038, 1664, 1146, 3619, 2070, 45, 368, 3067, 1354, 2825, 3999, 809, 729, 500, 371, 3290, 102, 3194, 985, 814, 3172, 2718, 2780, 446, 2633, 1473, 2881, 2048, 2949, 3253, 1661, 1183, 3051, 1468, 662, 3674, 805, 1656, 3814, 795, 3427, 3840, 189, 3151, 378, 1637, 3866, 747, 1091, 1261, 3662, 1334, 1863, 3700, 3577, 750, 3072, 1819, 712, 3571, 3024, 1392, 1620, 2801, 3324, 988, 3830, 1174, 3257, 1626, 3539, 1636, 3567, 1686, 1688, 2043, 2022, 1972, 544, 2036, 3981, 2578, 857, 884, 555, 3526, 158, 66, 2768, 1648, 1269, 3168, 2340, 3506, 3863, 328, 213, 3514, 1861, 3241, 304, 1090, 2579, 2254, 3247, 2520, 178, 2896, 1115, 1592, 1841, 2290, 1041, 3449, 455, 2998, 1045, 2180, 511, 3268, 2447, 3310, 1335, 309, 6, 1007, 3057, 1489, 2736, 883, 2759, 824, 3966, 3045, 2928, 2275, 1619, 856, 221, 443, 3818, 1296, 3334, 406, 2192, 827, 3246, 174, 4085, 3173, 1752, 4055, 2572, 2333, 2846, 1877, 4076, 3825, 2952, 2643, 505, 1073, 596, 4047, 413, 3251, 967, 156, 2993, 466, 237, 956, 968, 460, 2400, 2639, 2658, 70, 3348, 3586, 2187, 2240, 1171, 3289, 1046, 2232, 1748, 1481, 3092, 1856, 3842, 1113, 377, 2114, 2760, 2893, 487, 2295, 1176, 2772, 1998, 21, 3939, 2765], ('v_proj', 23): [2533, 3431, 2789, 1076, 2235, 3209, 2298, 2158, 1512, 3135, 2393, 4071, 3471, 1404, 363, 2944, 575, 257, 3994, 490, 462, 580, 1793, 3893, 2927, 2608, 2883, 2469, 94, 2958, 1218, 339, 2863, 2350, 2622, 3656, 788, 1571, 2050, 3852, 2077, 3946, 1307, 588, 2459, 375, 310, 1524, 1163, 239, 1868, 1057, 3526, 2358, 2281, 2611, 886, 1489, 3117, 3536, 972, 1316, 3514, 866, 200, 3491, 2248, 2736, 4063, 3921, 3872, 1167, 1523, 1930, 487, 134, 2168, 3674, 1378, 707, 2016, 685, 739, 1119, 1548, 1721, 299, 2949, 3106, 3979, 1825, 1583, 3273, 883, 2801, 956, 2573, 3249, 2740, 3290, 1360, 3562, 2688, 3840, 2187, 186, 3148, 968, 2433, 1117, 2522, 3391, 1312, 3999, 2147, 1134, 1449, 1409, 2750, 2476, 1734, 2199, 3816, 2204, 345, 1619, 3629, 2825, 4085, 2581, 2403, 378, 2192, 3065, 3807, 3836, 3051, 2377, 1551, 1180, 3854, 2412, 3556, 2308, 1399, 3031, 745, 2140, 2988, 1510, 3722, 3245, 1062, 368, 851, 1545, 3731, 1272, 2718, 1374, 340, 3952, 2643, 2869, 1888, 2959, 1053, 837, 750, 1492, 2513, 130, 1354, 2687, 2742, 150, 584, 1114, 1648, 2282, 1334, 3333, 3772, 3, 712, 3636, 3579, 2310, 3767, 666, 655, 2815, 3362, 2540, 338, 2226, 446, 3080, 1025, 305, 3863, 3233, 1269, 3703, 3512, 949, 309, 3371, 1772, 261, 2654, 1174, 66], ('q_proj', 23): [2533, 3431, 2789, 1076, 2235, 3209, 2298, 2158, 1512, 3135, 2393, 4071, 3471, 1404, 363, 2944, 575, 257, 3994, 490, 462, 580, 1793, 3893, 2927, 2608, 2883, 2469, 94, 2958, 1218, 339, 2863, 2350, 2622, 3656, 788, 1571, 2050, 3852, 2077, 3946, 1307, 588, 2459, 375, 310, 1524, 1163, 239, 1868, 1057, 3526, 2358, 2281, 2611, 886, 1489, 3117, 3536, 972, 1316, 3514, 866, 200, 3491, 2248, 2736, 4063, 3921, 3872, 1167, 1523, 1930, 487, 134, 2168, 3674, 1378, 707, 2016, 685, 739, 1119, 1548, 1721, 299, 2949, 3106, 3979, 1825, 1583, 3273, 883, 2801, 956, 2573, 3249, 2740, 3290, 1360, 3562, 2688, 3840, 2187, 186, 3148, 968, 2433, 1117, 2522, 3391, 1312, 3999, 2147, 1134, 1449, 1409, 2750, 2476, 1734, 2199, 3816, 2204, 345, 1619, 3629, 2825, 4085, 2581, 2403, 378, 2192, 3065, 3807, 3836, 3051, 2377, 1551, 1180, 3854, 2412, 3556, 2308, 1399, 3031, 745, 2140, 2988, 1510, 3722, 3245, 1062, 368, 851, 1545, 3731, 1272, 2718, 1374, 340, 3952, 2643, 2869, 1888, 2959, 1053, 837, 750, 1492, 2513, 130, 1354, 2687, 2742, 150, 584, 1114, 1648, 2282, 1334, 3333, 3772, 3, 712, 3636, 3579, 2310, 3767, 666, 655, 2815, 3362, 2540, 338, 2226, 446, 3080, 1025, 305, 3863, 3233, 1269, 3703, 3512, 949, 309, 3371, 1772, 261, 2654, 1174], ('k_proj', 23): [2533, 3431, 2789, 1076, 2235, 3209, 2298, 2158, 1512, 3135, 2393, 4071, 3471, 1404, 363, 2944, 575, 257, 3994, 490, 462, 580, 1793, 3893, 2927, 2608, 2883, 2469, 94, 2958, 1218, 339, 2863, 2350, 2622, 3656, 788, 1571, 2050, 3852, 2077, 3946, 1307, 588, 2459, 375, 310, 1524, 1163, 239, 1868, 1057, 3526, 2358, 2281, 2611, 886, 1489, 3117, 3536, 972, 1316, 3514, 866, 200, 3491, 2248, 2736, 4063, 3921, 3872, 1167, 1523, 1930, 487, 134, 2168, 3674, 1378, 707, 2016, 685, 739, 1119, 1548, 1721, 299, 2949, 3106, 3979, 1825, 1583, 3273, 883, 2801, 956, 2573, 3249, 2740, 3290, 1360, 3562, 2688, 3840, 2187, 186, 3148, 968, 2433, 1117, 2522, 3391, 1312, 3999, 2147, 1134, 1449, 1409, 2750, 2476, 1734, 2199, 3816, 2204, 345, 1619, 3629, 2825, 4085, 2581, 2403, 378, 2192, 3065, 3807, 3836, 3051, 2377, 1551, 1180, 3854, 2412, 3556, 2308, 1399, 3031, 745, 2140, 2988, 1510, 3722, 3245, 1062, 368, 851, 1545, 3731, 1272, 2718, 1374, 340, 3952, 2643, 2869, 1888, 2959, 1053, 837, 750, 1492, 2513, 130, 1354, 2687, 2742, 150, 584, 1114, 1648, 2282, 1334, 3333, 3772, 3, 712, 3636, 3579, 2310, 3767, 666, 655, 2815, 3362, 2540, 338, 2226, 446, 3080, 1025, 305, 3863, 3233, 1269, 3703, 3512, 949, 309, 3371, 1772, 261, 2654, 1174], ('v_proj', 21): [2533, 2789, 3431, 1076, 2298, 3209, 2235, 2158, 2393, 1512, 3135, 3471, 4071, 1404, 257, 2944, 363, 490, 575, 2927, 3994, 462, 94, 1218, 580, 3893, 2350, 2608, 2469, 2863, 339, 1793, 2958, 1571, 788, 3656, 2622, 588, 2050, 3852, 3946, 1307, 2883, 2459, 2077, 375, 1524, 310, 3872, 2168, 972, 1868, 1163, 2358, 239, 3526, 2611, 3117, 2736, 3979, 1057, 345, 1489, 1523, 340, 1548, 2573, 1119, 886, 299, 1583, 186, 851, 866, 2281, 1930, 134, 3536, 1167, 2412, 3862, 707, 1316, 1510, 3921, 3273, 968, 3031, 3491, 2204, 685, 1619, 130, 3514, 3556, 883, 2140, 309, 3922, 1374, 2389, 3840, 2147, 712, 487, 749, 3106, 1151], ('q_proj', 21): [2533, 2789, 3431, 1076, 2298, 3209, 2235, 2158, 2393, 1512, 3135, 3471, 4071, 1404, 257, 2944, 363, 490, 575, 2927, 3994, 462, 94, 1218, 580, 3893, 2350, 2608, 2469, 2863, 339, 1793, 2958, 1571, 788, 3656, 2622, 588, 2050, 3852, 3946, 1307, 2883, 2459, 2077, 375, 1524, 310, 3872, 2168, 972, 1868, 1163, 2358, 239, 3526, 2611, 3117, 2736, 3979, 1057, 345, 1489, 1523, 340, 1548, 2573, 1119, 886, 299, 1583, 186, 851, 866, 2281, 1930, 134, 3536, 1167, 2412, 3862, 707, 1316, 1510, 3921, 3273, 968, 3031, 3491, 2204, 685, 1619, 130, 3514, 3556, 883, 2140, 309, 3922, 1374, 2389, 3840, 2147, 712, 487, 749, 3106, 1151], ('k_proj', 21): [2533, 2789, 3431, 1076, 2298, 3209, 2235, 2158, 2393, 1512, 3135, 3471, 4071, 1404, 257, 2944, 363, 490, 575, 2927, 3994, 462, 94, 1218, 580, 3893, 2350, 2608, 2469, 2863, 339, 1793, 2958, 1571, 788, 3656, 2622, 588, 2050, 3852, 3946, 1307, 2883, 2459, 2077, 375, 1524, 310, 3872, 2168, 972, 1868, 1163, 2358, 239, 3526, 2611, 3117, 2736, 3979, 1057, 345, 1489, 1523, 340, 1548, 2573, 1119, 886, 299, 1583, 186, 851, 866, 2281, 1930, 134, 3536, 1167, 2412, 3862, 707, 1316, 1510, 3921, 3273, 968, 3031, 3491, 2204, 685, 1619, 130, 3514, 3556, 883, 2140, 309, 3922, 1374, 2389, 3840, 2147, 712, 487, 749, 3106, 1151], ('v_proj', 9): [2533, 1076, 3431, 339, 2789, 2158, 2350, 3209, 2393, 2298, 257, 1512, 4071, 2235, 1404, 490, 3135, 94, 2469, 3994, 363, 2608, 1218, 1571, 2168, 462, 2927, 1793, 2944, 3999, 3656, 2459, 588, 580, 2958, 575, 788, 2147, 3893, 2050, 2016, 474, 2622, 2863, 1583, 3491, 310, 1272], ('q_proj', 9): [2533, 1076, 3431, 339, 2789, 2158, 2350, 3209, 2393, 2298, 257, 1512, 4071, 2235, 1404, 490, 3135, 94, 2469, 3994, 363, 2608, 1218, 1571, 2168, 462, 2927, 1793, 2944, 3999, 3656, 2459, 588, 580, 2958, 575, 788, 2147, 3893, 2050, 2016, 474, 2622, 2863, 1583, 3491, 310, 1272], ('k_proj', 9): [2533, 1076, 3431, 339, 2789, 2158, 2350, 3209, 2393, 2298, 257, 1512, 4071, 2235, 1404, 490, 3135, 94, 2469, 3994, 363, 2608, 1218, 1571, 2168, 462, 2927, 1793, 2944, 3999, 3656, 2459, 588, 580, 2958, 575, 788, 2147, 3893, 2050, 2016, 474, 2622, 2863, 1583, 3491, 310, 1272], ('v_proj', 30): [2533, 1512, 1076, 3431, 2789, 2158, 2298, 3209, 3135, 2944, 2393, 2235, 363, 575, 257, 4071, 3994, 580, 2883, 1404, 3893, 462, 2469, 1793, 339, 1218, 2350, 2958, 2622, 3656, 2608, 2927, 1571, 2281, 490, 2050, 788, 3471, 3852, 2077, 1307, 1360, 4053, 1167, 239, 3772, 1721, 358, 3270, 3946, 1734, 375, 2639, 3952, 588, 1335, 1134, 2279, 2358, 2333, 3065, 988, 666, 2627, 94, 866, 2459, 2551, 367, 824, 3771, 3807, 745, 2095, 3571, 3391, 3839, 2204, 803, 2324, 1397, 3067, 747, 4076, 3830, 2018, 2403, 568, 3148, 1844, 3402, 1117, 45, 3558, 2654, 1524, 1062, 188, 2625, 3445, 3556, 470, 3205, 805, 739, 1863, 3494, 3072, 3591, 3249, 2326, 707, 1316, 2825, 1354, 3804, 1425, 2750, 102, 4082, 3674, 178, 1763, 1825, 2131, 1755, 2658, 1857, 2924, 596, 2852, 2805, 361, 1545, 3178, 1635, 3512, 1626, 1017, 1930, 2217, 2573, 1944, 2180, 3819, 1051, 3836, 3921, 1664, 1392, 3334, 158, 1409, 1761, 3241, 4063, 3037, 1661, 669, 1461, 2042, 2803, 2801, 3703, 658, 1176, 2476, 1748, 3132, 972, 1747, 809, 3290, 3579, 1468, 3465, 2254, 2869, 2963, 2934, 1334, 1501, 1478, 3717, 1551, 3435, 2939, 2543, 3427, 2815, 3310, 2772, 2443, 1075, 2998, 2433, 4085, 2718, 2763, 2232, 2540, 1868, 1244, 2275, 1472, 2410, 1731, 2602, 2839, 3244, 780, 2880, 856, 1073, 3814, 3629, 2579, 1802, 3319, 1296, 1163, 21, 3276, 4055, 3092, 975, 2571, 3653, 237, 3296, 2428, 3335, 1877, 1549, 3449, 1103, 556, 2617, 397, 3619, 3842, 2130, 1698, 3768, 1330, 3700, 1508, 3123, 204, 1119, 2644, 1577, 1449, 2126, 1998, 299, 351, 134, 2203, 3803, 2295, 1819, 1548, 156, 413, 2964, 2290, 2026, 1272, 3837, 2016, 1952, 1146, 3247, 1580, 3710, 3181, 3825, 1091, 200, 3990, 149, 790, 819, 1374, 1054, 513, 1888, 2288, 656, 3602, 3995, 2390, 3679, 3386, 3169, 913, 1130, 2365, 28, 1752, 2826, 2513, 2701, 2237, 1115, 2598, 3109, 654, 2024, 1204, 3704, 3562, 1174, 2420, 352, 3387, 3948, 949, 3316, 108, 2695, 1896, 1966, 3684, 1077, 1357, 1124, 443, 2070, 729, 2915, 2896, 1356, 3971, 888, 2248, 802, 448, 814, 1069, 3539, 406, 388, 3246, 3328, 1261, 309, 2665, 2375, 3976, 1088, 279, 1223, 795, 2052, 3394, 216, 1285, 555, 2881, 3257, 3390, 2160, 252, 3245, 3080, 2400, 2200, 101, 3873, 2578, 3840, 1057, 1523, 770, 4018, 2226, 3731, 3816, 3095, 2272, 1344, 1653, 3444, 1936, 3640, 3877, 3151, 1911, 2897, 807, 2688, 199, 1495, 859, 1473, 306, 148, 3668, 316, 2439, 1656, 93, 1068, 500, 1927, 731, 2867, 530, 1686, 1183, 3919, 3061, 1629, 844, 1615, 2043, 150, 2199, 221, 2520, 3075, 207, 1754, 560, 372, 1025, 1665, 3834, 3027, 1280, 3186, 1041, 3609, 3362, 1308, 3809, 368, 886, 2561, 1492, 2684, 2022, 155, 213, 53, 446, 3577, 3164, 2150, 540, 310, 2523, 1569, 3701, 3184, 2591, 2140, 924, 1531, 3726, 1939, 2244, 2048, 589, 1254, 1772, 478, 787, 91, 3060, 440, 1239, 1841, 3500, 2457, 2611, 3119, 222, 661, 978, 2381, 879, 2581, 2728, 7, 2171, 3735, 2741, 3383, 700, 195, 417, 911, 2761, 1121, 3020, 2322, 1489, 3959, 2215, 2089, 1688, 3536, 1046, 619, 350, 2056, 970, 3662, 3042, 1009, 1045, 1768, 2797, 2282, 274, 3694, 493, 66, 3614, 2780, 2259, 70, 1481, 2847, 689, 3348, 2633, 3918, 746, 1257, 1349, 3506, 2921, 1592, 2714, 3287, 592, 1948, 3064, 2544, 2038, 229, 2354, 2907, 35, 1856, 1034, 3127, 786, 2015, 3654, 695, 16, 1751, 1399, 571, 503, 2659, 1179, 1986, 3268, 616, 2318, 3172, 2341, 3175, 894, 2399, 4017, 1620, 2303, 1504, 2447, 1089, 3722, 587, 3480, 1452, 103, 3032, 1800, 1690, 3898, 1978, 2680, 655, 3756, 2861, 3324, 1787, 2760, 2879, 2558, 371, 1291, 884, 2487, 2441, 3, 3207, 3054, 2612, 1660, 1517, 3045, 3430, 2484, 505, 1479, 2863, 1395, 2703, 2975, 1576, 3854, 1457, 2236, 116, 701, 3565, 3173, 3367, 3773, 1575, 3564, 3620, 3305, 44, 3120, 818, 696, 3400, 3760, 1750, 1333, 1991, 2402, 985, 2240, 2949, 2698, 1861, 1094, 2968, 2419, 1915, 3875, 2194, 3156, 1566, 2534, 2600, 857, 2168, 3066, 546, 1312, 3338, 3497, 1808, 2771, 1586, 2195, 952, 3843, 2490, 531, 1127, 3541, 1315, 3447, 2391, 3174, 2023, 538, 3103, 1809, 1171, 1097, 260, 2640, 2812, 2584, 3745, 2177, 826, 3966, 1824, 6, 1427, 1311, 3243, 3749, 2642, 2397, 2340, 1012, 1085, 2648, 2846, 1999, 3936, 2842, 1147, 2809, 3592, 3414, 2031, 3442, 1740, 1156, 3483, 2807, 75, 968, 2765, 4068, 268, 1053, 2431, 659, 3582, 3977, 873, 1297, 1138, 1499, 286, 1230, 3333, 1781, 4081, 1310, 1407, 2405, 2005, 3273, 880, 3388, 1960, 11, 2053, 3231, 3293, 3953, 451, 951, 3210, 2239, 2747, 3485, 1574, 2494, 2961, 3886, 189, 62, 954, 292, 3300, 409, 1180, 955, 3448, 3569, 1745, 3024, 415, 3466, 2445, 2411, 648, 1189, 2084, 4090, 3117, 3331, 712, 140, 2713, 1847, 2061, 2755, 1510, 1920, 1135, 2205, 2343, 1262, 112, 1666], ('q_proj', 30): [2533, 1512, 1076, 3431, 2789, 2158, 2298, 3209, 3135, 2944, 2393, 2235, 363, 575, 257, 4071, 3994, 580, 2883, 1404, 3893, 462, 2469, 1793, 339, 1218, 2350, 2958, 2622, 3656, 2608, 2927, 1571, 2281, 490, 2050, 788, 3471, 3852, 2077, 1307, 1360, 4053, 1167, 239, 3772, 1721, 358, 3270, 3946, 1734, 375, 2639, 3952, 588, 1335, 1134, 2279, 2358, 2333, 3065, 988, 666, 2627, 94, 866, 2459, 2551, 367, 824, 3771, 3807, 745, 2095, 3571, 3391, 3839, 2204, 803, 2324, 1397, 3067, 747, 4076, 3830, 2018, 2403, 568, 3148, 1844, 3402, 1117, 45, 3558, 2654, 1524, 1062, 188, 2625, 3445, 3556, 470, 3205, 805, 739, 1863, 3494, 3072, 3591, 3249, 2326, 707, 1316, 2825, 1354, 3804, 1425, 2750, 102, 4082, 3674, 178, 1763, 1825, 2131, 1755, 2658, 1857, 2924, 596, 2852, 2805, 361, 1545, 3178, 1635, 3512, 1626, 1017, 1930, 2217, 2573, 1944, 2180, 3819, 1051, 3836, 3921, 1664, 1392, 3334, 158, 1409, 1761, 3241, 4063, 3037, 1661, 669, 1461, 2042, 2803, 2801, 3703, 658, 1176, 2476, 1748, 3132, 972, 1747, 809, 3290, 3579, 1468, 3465, 2254, 2869, 2963, 2934, 1334, 1501, 1478, 3717, 1551, 3435, 2939, 2543, 3427, 2815, 3310, 2772, 2443, 1075, 2998, 2433, 4085, 2718, 2763, 2232, 2540, 1868, 1244, 2275, 1472, 2410, 1731, 2602, 2839, 3244, 780, 2880, 856, 1073, 3814, 3629, 2579, 1802, 3319, 1296, 1163, 21, 3276, 4055, 3092, 975, 2571, 3653, 237, 3296, 2428, 3335, 1877, 1549, 3449, 1103, 556, 2617, 397, 3619, 3842, 2130, 1698, 3768, 1330, 3700, 1508, 3123, 204, 1119, 2644, 1577, 1449, 2126, 1998, 299, 351, 134, 2203, 3803, 2295, 1819, 1548, 156, 413, 2964, 2290, 2026, 1272, 3837, 2016, 1952, 1146, 3247, 1580, 3710, 3181, 3825, 1091, 200, 3990, 149, 790, 819, 1374, 1054, 513, 1888, 2288, 656, 3602, 3995, 2390, 3679, 3386, 3169, 913, 1130, 2365, 28, 1752, 2826, 2513, 2701, 2237, 1115, 2598, 3109, 654, 2024, 1204, 3704, 3562, 1174, 2420, 352, 3387, 3948, 949, 3316, 108, 2695, 1896, 1966, 3684, 1077, 1357, 1124, 443, 2070, 729, 2915, 2896, 1356, 3971, 888, 2248, 802, 448, 814, 1069, 3539, 406, 388, 3246, 3328, 1261, 309, 2665, 2375, 3976, 1088, 279, 1223, 795, 2052, 3394, 216, 1285, 555, 2881, 3257, 3390, 2160, 252, 3245, 3080, 2400, 2200, 101, 3873, 2578, 3840, 1057, 1523, 770, 4018, 2226, 3731, 3816, 3095, 2272, 1344, 1653, 3444, 1936, 3640, 3877, 3151, 1911, 2897, 807, 2688, 199, 1495, 859, 1473, 306, 148, 3668, 316, 2439, 1656, 93, 1068, 500, 1927, 731, 2867, 530, 1686, 1183, 3919, 3061, 1629, 844, 1615, 2043, 150, 2199, 221, 2520, 3075, 207, 1754, 560, 372, 1025, 1665, 3834, 3027, 1280, 3186, 1041, 3609, 3362, 1308, 3809, 368, 886, 2561, 1492, 2684, 2022, 155, 213, 53, 446, 3577, 3164, 2150, 540, 310, 2523, 1569, 3701, 3184, 2591, 2140, 924, 1531, 3726, 1939, 2244, 2048, 589, 1254, 1772, 478, 787, 91, 3060, 440, 1239, 1841, 3500, 2457, 2611, 3119, 222, 661, 978, 2381, 879, 2581, 2728, 7, 2171, 3735, 2741, 3383, 700, 195, 417, 911, 2761, 1121, 3020, 2322, 1489, 3959, 2215, 2089, 1688, 3536, 1046, 619, 350, 2056, 970, 3662, 3042, 1009, 1045, 1768, 2797, 2282, 274, 3694, 493, 66, 3614, 2780, 2259, 70, 1481, 2847, 689, 3348, 2633, 3918, 746, 1257, 1349, 3506, 2921, 1592, 2714, 3287, 592, 1948, 3064, 2544, 2038, 229, 2354, 2907, 35, 1856, 1034, 3127, 786, 2015, 3654, 695, 16, 1751, 1399, 571, 503, 2659, 1179, 1986, 3268, 616, 2318, 3172, 2341, 3175, 894, 2399, 4017, 1620, 2303, 1504, 2447, 1089, 3722, 587, 3480, 1452, 103, 3032, 1800, 1690, 3898, 1978, 2680, 655, 3756, 2861, 3324, 1787, 2760, 2879, 2558, 371, 1291, 884, 2487, 2441, 3, 3207, 3054, 2612, 1660, 1517, 3045, 3430, 2484, 505, 1479, 2863, 1395, 2703, 2975, 1576, 3854, 1457, 2236, 116, 701, 3565, 3173, 3367, 3773, 1575, 3564, 3620, 3305, 44, 3120, 818, 696, 3400, 3760, 1750, 1333, 1991, 2402, 985, 2240, 2949, 2698, 1861, 1094, 2968, 2419, 1915, 3875, 2194, 3156, 1566, 2534, 2600, 857, 2168, 3066, 546, 1312, 3338, 3497, 1808, 2771, 1586, 2195, 952, 3843, 2490, 531, 1127, 3541, 1315, 3447, 2391, 3174, 2023, 538, 3103, 1809, 1171, 1097, 260, 2640, 2812, 2584, 3745, 2177, 826, 3966, 1824, 6, 1427, 1311, 3243, 3749, 2642, 2397, 2340, 1012, 1085, 2648, 2846, 1999, 3936, 2842, 1147, 2809, 3592, 3414, 2031, 3442, 1740, 1156, 3483, 2807, 75, 968, 2765, 4068, 268, 1053, 2431, 659, 3582, 3977, 873, 1297, 1138, 1499, 286, 1230, 3333, 1781, 4081, 1310, 1407, 2405, 2005, 3273, 880, 3388, 1960, 11, 2053, 3231, 3293, 3953, 451, 951, 3210, 2239, 2747, 3485, 1574, 2494, 2961, 3886, 189, 62, 954, 292, 3300, 409, 1180, 955, 3448, 3569, 1745, 3024, 415, 3466, 2445, 2411, 648, 1189, 2084, 4090, 3117, 3331, 712, 140, 2713, 1847, 2061, 2755, 1510, 1920, 1135, 2205, 2343, 1262, 112, 1666], ('k_proj', 30): [2533, 1512, 1076, 3431, 2789, 2158, 2298, 3209, 3135, 2944, 2393, 2235, 363, 575, 257, 4071, 3994, 580, 2883, 1404, 3893, 462, 2469, 1793, 339, 1218, 2350, 2958, 2622, 3656, 2608, 2927, 1571, 2281, 490, 2050, 788, 3471, 3852, 2077, 1307, 1360, 4053, 1167, 239, 3772, 1721, 358, 3270, 3946, 1734, 375, 2639, 3952, 588, 1335, 1134, 2279, 2358, 2333, 3065, 988, 666, 2627, 94, 866, 2459, 2551, 367, 824, 3771, 3807, 745, 2095, 3571, 3391, 3839, 2204, 803, 2324, 1397, 3067, 747, 4076, 3830, 2018, 2403, 568, 3148, 1844, 3402, 1117, 45, 3558, 2654, 1524, 1062, 188, 2625, 3445, 3556, 470, 3205, 805, 739, 1863, 3494, 3072, 3591, 3249, 2326, 707, 1316, 2825, 1354, 3804, 1425, 2750, 102, 4082, 3674, 178, 1763, 1825, 2131, 1755, 2658, 1857, 2924, 596, 2852, 2805, 361, 1545, 3178, 1635, 3512, 1626, 1017, 1930, 2217, 2573, 1944, 2180, 3819, 1051, 3836, 3921, 1664, 1392, 3334, 158, 1409, 1761, 3241, 4063, 3037, 1661, 669, 1461, 2042, 2803, 2801, 3703, 658, 1176, 2476, 1748, 3132, 972, 1747, 809, 3290, 3579, 1468, 3465, 2254, 2869, 2963, 2934, 1334, 1501, 1478, 3717, 1551, 3435, 2939, 2543, 3427, 2815, 3310, 2772, 2443, 1075, 2998, 2433, 4085, 2718, 2763, 2232, 2540, 1868, 1244, 2275, 1472, 2410, 1731, 2602, 2839, 3244, 780, 2880, 856, 1073, 3814, 3629, 2579, 1802, 3319, 1296, 1163, 21, 3276, 4055, 3092, 975, 2571, 3653, 237, 3296, 2428, 3335, 1877, 1549, 3449, 1103, 556, 2617, 397, 3619, 3842, 2130, 1698, 3768, 1330, 3700, 1508, 3123, 204, 1119, 2644, 1577, 1449, 2126, 1998, 299, 351, 134, 2203, 3803, 2295, 1819, 1548, 156, 413, 2964, 2290, 2026, 1272, 3837, 2016, 1952, 1146, 3247, 1580, 3710, 3181, 3825, 1091, 200, 3990, 149, 790, 819, 1374, 1054, 513, 1888, 2288, 656, 3602, 3995, 2390, 3679, 3386, 3169, 913, 1130, 2365, 28, 1752, 2826, 2513, 2701, 2237, 1115, 2598, 3109, 654, 2024, 1204, 3704, 3562, 1174, 2420, 352, 3387, 3948, 949, 3316, 108, 2695, 1896, 1966, 3684, 1077, 1357, 1124, 443, 2070, 729, 2915, 2896, 1356, 3971, 888, 2248, 802, 448, 814, 1069, 3539, 406, 388, 3246, 3328, 1261, 309, 2665, 2375, 3976, 1088, 279, 1223, 795, 2052, 3394, 216, 1285, 555, 2881, 3257, 3390, 2160, 252, 3245, 3080, 2400, 2200, 101, 3873, 2578, 3840, 1057, 1523, 770, 4018, 2226, 3731, 3816, 3095, 2272, 1344, 1653, 3444, 1936, 3640, 3877, 3151, 1911, 2897, 807, 2688, 199, 1495, 859, 1473, 306, 148, 3668, 316, 2439, 1656, 93, 1068, 500, 1927, 731, 2867, 530, 1686, 1183, 3919, 3061, 1629, 844, 1615, 2043, 150, 2199, 221, 2520, 3075, 207, 1754, 560, 372, 1025, 1665, 3834, 3027, 1280, 3186, 1041, 3609, 3362, 1308, 3809, 368, 886, 2561, 1492, 2684, 2022, 155, 213, 53, 446, 3577, 3164, 2150, 540, 310, 2523, 1569, 3701, 3184, 2591, 2140, 924, 1531, 3726, 1939, 2244, 2048, 589, 1254, 1772, 478, 787, 91, 3060, 440, 1239, 1841, 3500, 2457, 2611, 3119, 222, 661, 978, 2381, 879, 2581, 2728, 7, 2171, 3735, 2741, 3383, 700, 195, 417, 911, 2761, 1121, 3020, 2322, 1489, 3959, 2215, 2089, 1688, 3536, 1046, 619, 350, 2056, 970, 3662, 3042, 1009, 1045, 1768, 2797, 2282, 274, 3694, 493, 66, 3614, 2780, 2259, 70, 1481, 2847, 689, 3348, 2633, 3918, 746, 1257, 1349, 3506, 2921, 1592, 2714, 3287, 592, 1948, 3064, 2544, 2038, 229, 2354, 2907, 35, 1856, 1034, 3127, 786, 2015, 3654, 695, 16, 1751, 1399, 571, 503, 2659, 1179, 1986, 3268, 616, 2318, 3172, 2341, 3175, 894, 2399, 4017, 1620, 2303, 1504, 2447, 1089, 3722, 587, 3480, 1452, 103, 3032, 1800, 1690, 3898, 1978, 2680, 655, 3756, 2861, 3324, 1787, 2760, 2879, 2558, 371, 1291, 884, 2487, 2441, 3, 3207, 3054, 2612, 1660, 1517, 3045, 3430, 2484, 505, 1479, 2863, 1395, 2703, 2975, 1576, 3854, 1457, 2236, 116, 701, 3565, 3173, 3367, 3773, 1575, 3564, 3620, 3305, 44, 3120, 818, 696, 3400, 3760, 1750, 1333, 1991, 2402, 985, 2240, 2949, 2698, 1861, 1094, 2968, 2419, 1915, 3875, 2194, 3156, 1566, 2534, 2600, 857, 2168, 3066, 546, 1312, 3338, 3497, 1808, 2771, 1586, 2195, 952, 3843, 2490, 531, 1127, 3541, 1315, 3447, 2391, 3174, 2023, 538, 3103, 1809, 1171, 1097, 260, 2640, 2812, 2584, 3745, 2177, 826, 3966, 1824, 6, 1427, 1311, 3243, 3749, 2642, 2397, 2340, 1012, 1085, 2648, 2846, 1999, 3936, 2842, 1147, 2809, 3592, 3414, 2031, 3442, 1740, 1156, 3483, 2807, 75, 968, 2765, 4068, 268, 1053, 2431, 659, 3582, 3977, 873, 1297, 1138, 1499, 286, 1230, 3333, 1781, 4081, 1310, 1407, 2405, 2005, 3273, 880, 3388, 1960, 11, 2053, 3231, 3293, 3953, 451, 951, 3210, 2239, 2747, 3485, 1574, 2494, 2961, 3886, 189, 62, 954, 292, 3300, 409, 1180, 955, 3448, 3569, 1745, 3024, 415, 3466, 2445, 2411, 648, 1189, 2084, 4090, 3117, 3331, 712, 140, 2713, 1847, 2061, 2755, 1510, 1920, 1135, 2205, 2343, 1262, 112, 1666], ('v_proj', 8): [2533, 1076, 3431, 2789, 339, 2158, 2298, 3209, 2393, 2350, 1512, 4071, 257, 1404, 490, 3135, 94, 2235, 3994, 2469, 2168, 363, 1571, 2608, 1218, 462, 3656, 2944, 2927, 2459, 3999, 1793, 588, 580, 2958, 3893, 2147, 575, 2050, 788, 3946, 2622, 1307, 3807, 2016, 3491, 1272, 1583, 2863], ('q_proj', 8): [2533, 1076, 3431, 2789, 339, 2158, 2298, 3209, 2393, 2350, 1512, 4071, 257, 1404, 490, 3135, 94, 2235, 3994, 2469, 2168, 363, 1571, 2608, 1218, 462, 3656, 2944, 2927, 2459, 3999, 1793, 588, 580, 2958, 3893, 2147, 575, 2050, 788, 3946, 2622, 1307, 3807, 2016, 3491, 1272, 1583, 2863], ('k_proj', 8): [2533, 1076, 3431, 2789, 339, 2158, 2298, 3209, 2393, 2350, 1512, 4071, 257, 1404, 490, 3135, 94, 2235, 3994, 2469, 2168, 363, 1571, 2608, 1218, 462, 3656, 2944, 2927, 2459, 3999, 1793, 588, 580, 2958, 3893, 2147, 575, 2050, 788, 3946, 2622, 1307, 3807, 2016, 3491, 1272, 1583, 2863], ('v_proj', 29): [2533, 3431, 2789, 1076, 3209, 2158, 1512, 2298, 3135, 2235, 2393, 4071, 363, 2944, 1404, 257, 575, 2883, 3994, 339, 580, 462, 2469, 1793, 3893, 2350, 1218, 2622, 2608, 3471, 2958, 2927, 3656, 2050, 490, 2281, 94, 1307, 788, 2077, 1571, 3852, 3391, 4053, 2639, 239, 375, 1360, 1734, 588, 2459, 3946, 1721, 2627, 3771, 1335, 1167, 2403, 4063, 568, 2654, 2358, 2279, 972, 3772, 3270, 3065, 3807, 1825, 358, 1316, 470, 1134, 1524, 2095, 669, 866, 3952, 3067, 805, 3249, 2204, 747, 367, 1062, 1461, 3435, 3148, 2573, 988, 2924, 745, 4076, 1296, 2018, 3839, 3768, 2168, 790, 1017, 2625, 3444, 3571, 2551, 149, 2326, 1930, 3921, 2611, 4082, 1857, 2540, 2333, 1545, 803, 3830, 102, 2852, 2863, 2254, 3556, 3445, 1117, 2825, 1763, 3328, 3836, 1374, 1748, 1501, 1334, 3334, 2543, 188, 45, 729, 1449, 707, 3558, 824, 556, 1548, 3804, 3244, 3205, 1635, 1761, 3290, 2718, 3803, 3710, 2644, 2248, 1888, 2476, 1425, 1551, 3506, 1397, 4085, 1944, 2658, 2410, 252, 309, 3335, 1163, 4055, 3362, 3562, 3591, 1629, 3819, 2869, 2815, 3427, 2763, 2612, 3324, 3386, 3579, 3194, 3619, 299, 2695, 2750, 2839, 178, 2617, 310, 1468, 3465, 2275, 3310, 3629, 2688, 1844, 2934, 1863, 3241, 2975, 2803, 2217, 886, 3536, 1549, 1664, 1130, 2772, 3674, 1472, 2131, 3072, 1354, 2016, 388, 2226, 156, 2579, 1057, 3854, 856, 1868, 2026, 158, 2324, 666, 2578, 1261, 1688, 2022, 596, 3814, 1073, 3700, 2591, 3390, 739, 3169, 3402, 500, 2200, 200, 1054, 1119, 1068, 2070, 1090, 2126, 814, 1051, 1580, 4018, 1861, 1510, 1344, 3080, 1244, 3512, 1103, 2203, 3, 1566, 3132, 3816, 3837, 505, 2177, 589, 2760, 2443, 1481, 3037, 3679, 1138, 216, 1204, 1772, 1146, 2038, 2513, 189, 368, 1077, 2160, 2365, 1998, 3990, 1755, 3825, 3602, 2801, 1819, 2968, 1747, 2846, 213, 2290, 1393, 2561, 448, 2428, 3296, 2581, 3095, 985, 3151, 780, 1285, 3731, 1626, 134, 3662, 3045, 3653, 397, 3348, 1124, 3168, 795, 2375, 2633, 1115, 3172, 2085, 1802, 1174, 648, 3840, 2881, 2607, 3178, 2295, 3582, 101, 1399, 2180, 2964, 2998, 3918, 2921, 1075, 2354, 3092, 1569, 1176, 513, 3236, 3703, 1478, 1661, 2602, 1473, 3966, 818, 3778, 155, 2805, 1731, 3773, 2684, 3722, 3245, 819, 1392, 306, 1577, 1523, 2571, 2544, 3079, 2761, 560, 4070, 731, 2939, 2232, 108, 3577, 114, 1877, 857, 352, 1492, 1489, 2140, 1686, 809, 3498, 260, 3197, 3995, 2880, 2397, 3529, 2288, 3704, 3075, 1272, 2199, 2043, 530, 28, 66, 786, 3873, 3117, 2723, 1441, 689, 3123, 417, 2048, 493, 237, 1952, 351, 658, 2949, 446, 1127, 3319, 413, 1007, 3564, 148, 2061, 2701, 1637, 3717, 2411, 2150, 1089, 381, 544, 204, 949, 1926, 655, 3953, 2237, 2390, 2378, 2780, 2915, 2282, 443, 3442, 3354, 3181, 1088, 3640, 222, 1257, 21, 2433, 3184, 2052, 1991, 271, 1180, 1966, 3919, 2520, 685, 654, 3253, 1147, 3207, 3539, 2042, 700, 3834, 234, 3316, 2447, 2244, 826, 1409, 2676, 316, 1091, 2420, 3689, 1574, 1121, 7, 695, 802, 4046, 3756, 1583, 70, 245, 85, 911, 2640, 1009, 3449, 538, 662, 2141, 2774, 1936, 3305, 1280, 787, 1911, 2867, 2340, 3491, 1452, 1395, 2523, 317, 661, 2601, 1856, 371, 1025], ('q_proj', 29): [2533, 3431, 2789, 1076, 3209, 2158, 1512, 2298, 3135, 2235, 2393, 4071, 363, 2944, 1404, 257, 575, 2883, 3994, 339, 580, 462, 2469, 1793, 3893, 2350, 1218, 2622, 2608, 3471, 2958, 2927, 3656, 2050, 490, 2281, 94, 1307, 788, 2077, 1571, 3852, 3391, 4053, 2639, 239, 375, 1360, 1734, 588, 2459, 3946, 1721, 2627, 3771, 1335, 1167, 2403, 4063, 568, 2654, 2358, 2279, 972, 3772, 3270, 3065, 3807, 1825, 358, 1316, 470, 1134, 1524, 2095, 669, 866, 3952, 3067, 805, 3249, 2204, 747, 367, 1062, 1461, 3435, 3148, 2573, 988, 2924, 745, 4076, 1296, 2018, 3839, 3768, 2168, 790, 1017, 2625, 3444, 3571, 2551, 149, 2326, 1930, 3921, 2611, 4082, 1857, 2540, 2333, 1545, 803, 3830, 102, 2852, 2863, 2254, 3556, 3445, 1117, 2825, 1763, 3328, 3836, 1374, 1748, 1501, 1334, 3334, 2543, 188, 45, 729, 1449, 707, 3558, 824, 556, 1548, 3804, 3244, 3205, 1635, 1761, 3290, 2718, 3803, 3710, 2644, 2248, 1888, 2476, 1425, 1551, 3506, 1397, 4085, 1944, 2658, 2410, 252, 309, 3335, 1163, 4055, 3362, 3562, 3591, 1629, 3819, 2869, 2815, 3427, 2763, 2612, 3324, 3386, 3579, 3194, 3619, 299, 2695, 2750, 2839, 178, 2617, 310, 1468, 3465, 2275, 3310, 3629, 2688, 1844, 2934, 1863, 3241, 2975, 2803, 2217, 886, 3536, 1549, 1664, 1130, 2772, 3674, 1472, 2131, 3072, 1354, 2016, 388, 2226, 156, 2579, 1057, 3854, 856, 1868, 2026, 158, 2324, 666, 2578, 1261, 1688, 2022, 596, 3814, 1073, 3700, 2591, 3390, 739, 3169, 3402, 500, 2200, 200, 1054, 1119, 1068, 2070, 1090, 2126, 814, 1051, 1580, 4018, 1861, 1510, 1344, 3080, 1244, 3512, 1103, 2203, 3, 1566, 3132, 3816, 3837, 505, 2177, 589, 2760, 2443, 1481, 3037, 3679, 1138, 216, 1204, 1772, 1146, 2038, 2513, 189, 368, 1077, 2160, 2365, 1998, 3990, 1755, 3825, 3602, 2801, 1819, 2968, 1747, 2846, 213, 2290, 1393, 2561, 448, 2428, 3296, 2581, 3095, 985, 3151, 780, 1285, 3731, 1626, 134, 3662, 3045, 3653, 397, 3348, 1124, 3168, 795, 2375, 2633, 1115, 3172, 2085, 1802, 1174, 648, 3840, 2881, 2607, 3178, 2295, 3582, 101, 1399, 2180, 2964, 2998, 3918, 2921, 1075, 2354, 3092, 1569, 1176, 513, 3236, 3703, 1478, 1661, 2602, 1473, 3966, 818, 3778, 155, 2805, 1731, 3773, 2684, 3722, 3245, 819, 1392, 306, 1577, 1523, 2571, 2544, 3079, 2761, 560, 4070, 731, 2939, 2232, 108, 3577, 114, 1877, 857, 352, 1492, 1489, 2140, 1686, 809, 3498, 260, 3197, 3995, 2880, 2397, 3529, 2288, 3704, 3075, 1272, 2199, 2043, 530, 28, 66, 786, 3873, 3117, 2723, 1441, 689, 3123, 417, 2048, 493, 237, 1952, 351, 658, 2949, 446, 1127, 3319, 413, 1007, 3564, 148, 2061, 2701, 1637, 3717, 2411, 2150, 1089, 381, 544, 204, 949, 1926, 655, 3953, 2237, 2390, 2378, 2780, 2915, 2282, 443, 3442, 3354, 3181, 1088, 3640, 222, 1257, 21, 2433, 3184, 2052, 1991, 271, 1180, 1966, 3919, 2520, 685, 654, 3253, 1147, 3207, 3539, 2042, 700, 3834, 234, 3316, 2447, 2244, 826, 1409, 2676, 316, 1091, 2420, 3689, 1574, 1121, 7, 695, 802, 4046, 3756, 1583, 70, 245, 85, 911, 2640, 1009, 3449, 538, 662, 2141, 2774, 1936, 3305, 1280, 787, 1911, 2867, 2340, 3491, 1452, 1395, 2523, 317, 661, 2601, 1856, 371, 1025], ('k_proj', 29): [2533, 3431, 2789, 1076, 3209, 2158, 1512, 2298, 3135, 2235, 2393, 4071, 363, 2944, 1404, 257, 575, 2883, 3994, 339, 580, 462, 2469, 1793, 3893, 2350, 1218, 2622, 2608, 3471, 2958, 2927, 3656, 2050, 490, 2281, 94, 1307, 788, 2077, 1571, 3852, 3391, 4053, 2639, 239, 375, 1360, 1734, 588, 2459, 3946, 1721, 2627, 3771, 1335, 1167, 2403, 4063, 568, 2654, 2358, 2279, 972, 3772, 3270, 3065, 3807, 1825, 358, 1316, 470, 1134, 1524, 2095, 669, 866, 3952, 3067, 805, 3249, 2204, 747, 367, 1062, 1461, 3435, 3148, 2573, 988, 2924, 745, 4076, 1296, 2018, 3839, 3768, 2168, 790, 1017, 2625, 3444, 3571, 2551, 149, 2326, 1930, 3921, 2611, 4082, 1857, 2540, 2333, 1545, 803, 3830, 102, 2852, 2863, 2254, 3556, 3445, 1117, 2825, 1763, 3328, 3836, 1374, 1748, 1501, 1334, 3334, 2543, 188, 45, 729, 1449, 707, 3558, 824, 556, 1548, 3804, 3244, 3205, 1635, 1761, 3290, 2718, 3803, 3710, 2644, 2248, 1888, 2476, 1425, 1551, 3506, 1397, 4085, 1944, 2658, 2410, 252, 309, 3335, 1163, 4055, 3362, 3562, 3591, 1629, 3819, 2869, 2815, 3427, 2763, 2612, 3324, 3386, 3579, 3194, 3619, 299, 2695, 2750, 2839, 178, 2617, 310, 1468, 3465, 2275, 3310, 3629, 2688, 1844, 2934, 1863, 3241, 2975, 2803, 2217, 886, 3536, 1549, 1664, 1130, 2772, 3674, 1472, 2131, 3072, 1354, 2016, 388, 2226, 156, 2579, 1057, 3854, 856, 1868, 2026, 158, 2324, 666, 2578, 1261, 1688, 2022, 596, 3814, 1073, 3700, 2591, 3390, 739, 3169, 3402, 500, 2200, 200, 1054, 1119, 1068, 2070, 1090, 2126, 814, 1051, 1580, 4018, 1861, 1510, 1344, 3080, 1244, 3512, 1103, 2203, 3, 1566, 3132, 3816, 3837, 505, 2177, 589, 2760, 2443, 1481, 3037, 3679, 1138, 216, 1204, 1772, 1146, 2038, 2513, 189, 368, 1077, 2160, 2365, 1998, 3990, 1755, 3825, 3602, 2801, 1819, 2968, 1747, 2846, 213, 2290, 1393, 2561, 448, 2428, 3296, 2581, 3095, 985, 3151, 780, 1285, 3731, 1626, 134, 3662, 3045, 3653, 397, 3348, 1124, 3168, 795, 2375, 2633, 1115, 3172, 2085, 1802, 1174, 648, 3840, 2881, 2607, 3178, 2295, 3582, 101, 1399, 2180, 2964, 2998, 3918, 2921, 1075, 2354, 3092, 1569, 1176, 513, 3236, 3703, 1478, 1661, 2602, 1473, 3966, 818, 3778, 155, 2805, 1731, 3773, 2684, 3722, 3245, 819, 1392, 306, 1577, 1523, 2571, 2544, 3079, 2761, 560, 4070, 731, 2939, 2232, 108, 3577, 114, 1877, 857, 352, 1492, 1489, 2140, 1686, 809, 3498, 260, 3197, 3995, 2880, 2397, 3529, 2288, 3704, 3075, 1272, 2199, 2043, 530, 28, 66, 786, 3873, 3117, 2723, 1441, 689, 3123, 417, 2048, 493, 237, 1952, 351, 658, 2949, 446, 1127, 3319, 413, 1007, 3564, 148, 2061, 2701, 1637, 3717, 2411, 2150, 1089, 381, 544, 204, 949, 1926, 655, 3953, 2237, 2390, 2378, 2780, 2915, 2282, 443, 3442, 3354, 3181, 1088, 3640, 222, 1257, 21, 2433, 3184, 2052, 1991, 271, 1180, 1966, 3919, 2520, 685, 654, 3253, 1147, 3207, 3539, 2042, 700, 3834, 234, 3316, 2447, 2244, 826, 1409, 2676, 316, 1091, 2420, 3689, 1574, 1121, 7, 695, 802, 4046, 3756, 1583, 70, 245, 85, 911, 2640, 1009, 3449, 538, 662, 2141, 2774, 1936, 3305, 1280, 787, 1911, 2867, 2340, 3491, 1452, 1395, 2523, 317, 661, 2601, 1856, 371, 1025], ('v_proj', 7): [2533, 1076, 3431, 2789, 339, 3209, 2158, 2298, 3135, 4071, 2350, 1404, 257, 2393, 1512, 490, 94, 3994, 2469, 2235, 1571, 2168, 462, 363, 2944, 1218, 2608, 2459, 3656, 2927, 2958, 580, 3999, 588, 575, 1793, 3893, 2147, 2050, 3946, 3807, 1307, 2622, 3491, 474, 788, 2863, 1261, 2016, 1272], ('q_proj', 7): [2533, 1076, 3431, 2789, 339, 3209, 2158, 2298, 3135, 4071, 2350, 1404, 257, 2393, 1512, 490, 94, 3994, 2469, 2235, 1571, 2168, 462, 363, 2944, 1218, 2608, 2459, 3656, 2927, 2958, 580, 3999, 588, 575, 1793, 3893, 2147, 2050, 3946, 3807, 1307, 2622, 3491, 474, 788, 2863, 1261, 2016, 1272], ('k_proj', 7): [2533, 1076, 3431, 2789, 339, 3209, 2158, 2298, 3135, 4071, 2350, 1404, 257, 2393, 1512, 490, 94, 3994, 2469, 2235, 1571, 2168, 462, 363, 2944, 1218, 2608, 2459, 3656, 2927, 2958, 580, 3999, 588, 575, 1793, 3893, 2147, 2050, 3946, 3807, 1307, 2622, 3491, 474, 788, 2863, 1261, 2016, 1272], ('v_proj', 6): [1076, 2533, 3431, 2789, 3209, 339, 2158, 2298, 3135, 1404, 4071, 2393, 257, 1512, 94, 2350, 490, 2469, 3994, 1571, 2235, 2944, 363, 2608, 462, 1218, 575, 2459, 3656, 2958, 580, 2168, 2927, 2147, 3893, 2050, 588, 3999, 1793, 3807, 3946, 1307, 788, 2622, 1272, 1261], ('q_proj', 6): [1076, 2533, 3431, 2789, 3209, 339, 2158, 2298, 3135, 1404, 4071, 2393, 257, 1512, 94, 2350, 490, 2469, 3994, 1571, 2235, 2944, 363, 2608, 462, 1218, 575, 2459, 3656, 2958, 580, 2168, 2927, 2147, 3893, 2050, 588, 3999, 1793, 3807, 3946, 1307, 788, 2622, 1272, 1261], ('k_proj', 6): [1076, 2533, 3431, 2789, 3209, 339, 2158, 2298, 3135, 1404, 4071, 2393, 257, 1512, 94, 2350, 490, 2469, 3994, 1571, 2235, 2944, 363, 2608, 462, 1218, 575, 2459, 3656, 2958, 580, 2168, 2927, 2147, 3893, 2050, 588, 3999, 1793, 3807, 3946, 1307, 788, 2622, 1272, 1261], ('v_proj', 5): [1076, 3431, 2533, 3209, 2789, 339, 2158, 3135, 2298, 2393, 4071, 1404, 1512, 257, 575, 490, 2469, 94, 2944, 363, 2235, 1571, 3994, 2350, 2608, 580, 1218, 462, 3893, 2459, 2147, 2958, 2927, 3656, 3807, 588, 1793, 3999, 2050, 2168], ('q_proj', 5): [1076, 3431, 2533, 3209, 2789, 339, 2158, 3135, 2298, 2393, 4071, 1404, 1512, 257, 575, 490, 2469, 94, 2944, 363, 2235, 1571, 3994, 2350, 2608, 580, 1218, 462, 3893, 2459, 2147, 2958, 2927, 3656, 3807, 588, 1793, 3999, 2050, 2168], ('k_proj', 5): [1076, 3431, 2533, 3209, 2789, 339, 2158, 3135, 2298, 2393, 4071, 1404, 1512, 257, 575, 490, 2469, 94, 2944, 363, 2235, 1571, 3994, 2350, 2608, 580, 1218, 462, 3893, 2459, 2147, 2958, 2927, 3656, 3807, 588, 1793, 3999, 2050, 2168], ('v_proj', 4): [1076, 3431, 3209, 2533, 3135, 2789, 2158, 339, 1512, 2298, 2393, 4071, 575, 1404, 580, 2469, 257, 490, 363, 94, 3994, 2608, 3893, 1571, 2944, 1218, 462, 2459, 2958, 2147, 2927, 3807, 3656, 2050, 2235, 1793, 588, 3999, 2350], ('q_proj', 4): [1076, 3431, 3209, 2533, 3135, 2789, 2158, 339, 1512, 2298, 2393, 4071, 575, 1404, 580, 2469, 257, 490, 363, 94, 3994, 2608, 3893, 1571, 2944, 1218, 462, 2459, 2958, 2147, 2927, 3807, 3656, 2050, 2235, 1793, 588, 3999, 2350], ('k_proj', 4): [1076, 3431, 3209, 2533, 3135, 2789, 2158, 339, 1512, 2298, 2393, 4071, 575, 1404, 580, 2469, 257, 490, 363, 94, 3994, 2608, 3893, 1571, 2944, 1218, 462, 2459, 2958, 2147, 2927, 3807, 3656, 2050, 2235, 1793, 588, 3999, 2350], ('v_proj', 3): [1076, 3431, 3135, 2789, 3209, 2158, 339, 2298, 575, 580, 2533, 1512, 2393, 2469, 3994, 4071, 1404, 490, 257, 363, 3893, 1571, 94, 2608, 2944, 2958, 1218, 2459, 462, 2050, 3807, 3656, 2147, 2927, 3852, 1793, 2235, 375, 3946, 3999, 1307, 1261, 588, 2622], ('q_proj', 3): [1076, 3431, 3135, 2789, 3209, 2158, 339, 2298, 575, 580, 2533, 1512, 2393, 2469, 3994, 4071, 1404, 490, 257, 363, 3893, 1571, 94, 2608, 2944, 2958, 1218, 2459, 462, 2050, 3807, 3656, 2147, 2927, 3852, 1793, 2235, 375, 3946, 3999, 1307, 1261, 588, 2622], ('k_proj', 3): [1076, 3431, 3135, 2789, 3209, 2158, 339, 2298, 575, 580, 2533, 1512, 2393, 2469, 3994, 4071, 1404, 490, 257, 363, 3893, 1571, 94, 2608, 2944, 2958, 1218, 2459, 462, 2050, 3807, 3656, 2147, 2927, 3852, 1793, 2235, 375, 3946, 3999, 1307, 1261, 588, 2622], ('v_proj', 2): [3431, 1076, 2158, 1512, 2789, 2298, 3135, 3209, 580, 339, 575, 2393, 2469, 2533, 3994, 4071, 1404, 490, 363, 3893, 2944, 2958, 1571, 1218, 257, 2608, 2459, 94, 3807, 462, 2235, 2050, 2147, 2077], ('q_proj', 2): [3431, 1076, 2158, 1512, 2789, 2298, 3135, 3209, 580, 339, 575, 2393, 2469, 2533, 3994, 4071, 1404, 490, 363, 3893, 2944, 2958, 1571, 1218, 257, 2608, 2459, 94, 3807, 462, 2235, 2050, 2147, 2077], ('k_proj', 2): [3431, 1076, 2158, 1512, 2789, 2298, 3135, 3209, 580, 339, 575, 2393, 2469, 2533, 3994, 4071, 1404, 490, 363, 3893, 2944, 2958, 1571, 1218, 257, 2608, 2459, 94, 3807, 462, 2235, 2050, 2147, 2077], ('v_proj', 0): [3964, 1411, 22, 4051, 1744, 4030, 2543], ('q_proj', 0): [3964, 1411, 22, 4051, 1744, 4030, 2543], ('k_proj', 0): [3964, 1411, 22, 4051, 1744, 4030, 2543]}

                        print_rank_0(
                            f"apply channel sparsity to attention layer",
                            args.global_rank)
                        print_rank_0(
                            f"attention activation{attention_activation}",
                            args.global_rank)
                        selected_channel_attention = select_channel_based_on_activation(
                            attention_activation,
                            args.num_attention_channel,
                            selection_strategy=args.selection_strategy,
                            model=args.model_name_or_path)
                        print_rank_0(
                            f"selected attention channel: {selected_channel_attention}",
                            args.global_rank)
                        torch.distributed.barrier()
                        print(
                            f"selected attention channel before sync, test!!: {selected_channel_attention}"
                        )

                    if args.num_mlp_channel > 0:
                        # if torch.distributed.get_world_size() > 1:
                        #     attention_activation[key].to(device)
                        #     for key, value in activation.items():
                        #         deepspeed.comm.all_reduce(activation[key])
                        #         # torch.distributed.all_reduce(activation[key], op=torch.distributed.ReduceOp.SUM)
                        #     attention_activation[key].cpu()

                        selected_channel = select_channel_based_on_activation(
                            activation,
                            args.num_mlp_channel,
                            selection_strategy=args.selection_strategy,
                            calculate_strategy=args.calculate_strategy,
                            model=args.model_name_or_path)
                        print_rank_0(
                            f"selected mlp channel: {selected_channel}",
                            args.global_rank)
                        # set the freeze for the unselected heads
                        # TODO: Need to revise freeze functions

                    # save index list
                    selected_channel_attention = synchronize_index_list(
                        args, selected_channel_attention, args.global_rank)
                    print(
                        f"selected attention channel after sync, test!!!: {selected_channel_attention}"
                    )
                    model = freeze_unselected_channel_layer(
                        model.module, selected_channel,
                        selected_channel_attention)

                # convert selected mlp/attention to linear_matrix_sparsity
                print(
                    "===========================SYSTEM IMPLEMENTATION======================================"
                )
                model = convert_linear_layer_to_channel_sparsity(
                    model, selected_channel, selected_channel_attention)

                model = make_model_gradient_checkpointing_compatible(model)

                optimizer_grouped_parameters = get_optimizer_sparse_grouped_parameters(
                    model, args.w_decay, args.smt_lr)

                # AdamSparseOptimizer = DeepSpeedCPUMatrixSparisityAdam if args.offload else FusedMatrixSparseAdam
                # AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
                AdamOptimizer = FusedAdam

                del optimizer
                del lr_scheduler
                del warmup_grads
                del attention_warmup_grads

                # in SMT paper, all results are obtained under ft_learning_rate = smt_lr = 9.65e-6.
                # Further experiments: set smt_lr = 1e-4 can further improve the accuracy.
                new_optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                              lr=args.ft_learning_rate,
                                              betas=(0.95, 0.999))
                # betas=(0.9, 0.95))

                del optimizer_grouped_parameters

                new_lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=new_optimizer,
                    num_warmup_steps=args.smt_lr_warmup_steps,
                    num_training_steps=args.num_ft_epochs *
                    num_update_steps_per_epoch - current_step_count,
                )

                ds_config["zero_optimization"]["offload_param"][
                    "device"] = "none"
                ds_config["zero_optimization"]["offload_optimizer"][
                    "device"] = "none"
                model, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=model,
                    optimizer=new_optimizer,
                    args=args,
                    config=ds_config,
                    lr_scheduler=new_lr_scheduler)

                # print matrix sparsity trainable parameters
                total_num = sum(p.numel() for p in model.module.parameters())
                selected_num = sum(p.numel()
                                   for p in model.module.parameters()
                                   if p.requires_grad)
                print_rank_0(f"Number of Total parameters: {total_num}",
                             args.global_rank)
                rate = (selected_num / total_num) * 100
                print_rank_0(
                    f"Number of trainable parameters: {selected_num}, about{rate}% matrix sparsity parameters in the model are training",
                    args.global_rank)
                # break
            ################################### Activation Selection finish #################################

            # time record starts, smt transfer and gradient select not included in time since they only happens in warm up iterations
            if torch.distributed.get_rank() == 0 and step % 200 == 0:
                start = time.time()

            batch = to_device(batch, device)

            # New function: accumulate warm up MLP or Attention activation value
            # Hector's Note: channel MLP/Attention activation calculation.
            if args.channel_sparsity and current_step_count < args.full_ft_steps:
                model.eval()
                with torch.no_grad():
                    print(
                        "===================================================")
                    print("Activation Collection Starts")
                    # get layers
                    # layers = get_blocks(model)
                    layers = model.model.layers

                    inps = []
                    layer_kwargs = {}

                    # layers[0] = layers[0].cuda()
                    # move_embed(model, "cuda")

                    # get input and kwargs to layer 0
                    # with_kwargs is only supported in PyTorch 2.0
                    # use this Catcher hack for now
                    class Catcher(nn.Module):
                        def __init__(self, module):
                            super().__init__()
                            self.module = module

                        def forward(self, *inp, **kwargs):
                            inps.append(inp)
                            layer_kwargs.update(kwargs)
                            raise ValueError  # early exit to break later inference

                    # patch layer 0 to catch input and kwargs
                    layers[0] = Catcher(layers[0])
                    print("layers", layers)
                    print("layers[0]", layers[0])
                    try:
                        # model(samples.to(next(model.parameters()).device))
                        output = model(**batch, use_cache=False)

                    except ValueError:  # work with early exit
                        pass
                    # del samples
                    layers[0] = layers[0].module  # restore
                    print("inps:", inps)
                    print("len inps:", len(inps))

                    inps = inps[0]
                    print("inps[0]:", inps[0])
                    print("inps[0] shape:", inps[0].shape)

                    # layers[0] = layers[0].cpu()
                    # move_embed(model, "cpu")
                    gc.collect()
                    torch.cuda.empty_cache()

                    # solve layer by layer
                    for i in tqdm.tqdm(range(len(layers)),
                                       desc="Running AWQ..."):
                        print(f"=============== layer {i} =================")

                        layer = layers[i]
                        # layer = layer.cuda()
                        named_linears = get_named_linears(layer)

                        # firstly, get input features of all linear layers
                        def cache_input_hook(m, x, y, name, feat_dict_mlp,
                                             feat_dict_attention):
                            x = x[0].abs()

                            # need to consider whether work for single gpu!!!
                            torch.distributed.barrier()
                            deepspeed.comm.all_reduce(x, async_op=False)

                            x = x.detach().cpu().to(torch.float32)
                            # name
                            if 'mlp' in name and args.num_mlp_channel > 0:
                                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                                # activation[module_name].append(x)
                                if (module_name, i) not in feat_dict_mlp:
                                    feat_dict_mlp[(module_name, i)] = x
                                else:
                                    feat_dict_mlp[(module_name, i)] += x

                            if 'self_attn' in name and args.num_attention_channel > 0:
                                # print("=============== TEST =================")
                                # print("self_attn in ", name)
                                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                                if module_name is not None:
                                    if (module_name,
                                            i) not in feat_dict_attention:
                                        feat_dict_attention[(module_name,
                                                             i)] = x
                                    else:
                                        feat_dict_attention[(module_name,
                                                             i)] += x

                        handles = []
                        for name in named_linears:
                            handles.append(
                                named_linears[name].register_forward_hook(
                                    functools.partial(
                                        cache_input_hook,
                                        name=name,
                                        feat_dict_mlp=activation,
                                        feat_dict_attention=attention_activation
                                    )))
                        # inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
                        # get output as next layer's input
                        print("inps:", inps)
                        print("len inps:", len(inps))
                        print("inps[0]:", inps[0])
                        print("inps[0] shape:", inps[0].shape)
                        inps = layer(inps[0], **layer_kwargs)
                        # inps = layer(inps[0], **layer_kwargs)[0]
                        for h in handles:
                            h.remove()

                    print("mlp activation:", activation)
                    print("attention activation:", attention_activation)

                    # Clear GPU memory
                    torch.cuda.empty_cache()
                current_step_count += 1
                model.train()
                continue

            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)

            # New function: accumulate warm up MLP or Attention matrix block gradient
            # Hector's Note: matrix block MLP/Attention gradient calculation.
            if args.matrix_sparsity and current_step_count < args.full_ft_steps:
                # store gradients
                pattern = re.compile(r'model\.layers\.(\d+)\.')

                for name, param in model.module.named_parameters():
                    match = pattern.search(name)
                    layer_number = int(match.group(1)) if match else None
                    if 'mlp' in name and num_downsampled_mlp_blocks > 0:
                        grad = safe_get_full_grad(
                            param)  # (hidden_dim, head_dim)
                        module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'

                        #defaultdict(torch.float32)
                        if (module_name, layer_number) not in warmup_grads:
                            # warmup_grads[(module_name, layer_number)] = grad.detach().to(torch.float32)
                            warmup_grads[(
                                module_name,
                                layer_number)] = grad.detach().cpu().to(
                                    torch.float32)

                        else:
                            warmup_grads[(
                                module_name,
                                layer_number)] += grad.detach().cpu().to(
                                    torch.float32)
                            # warmup_grads[(module_name, layer_number)] += grad.detach().to(torch.float32)
                            del grad

                    if 'self_attn' in name and 'weight' in name and num_downsampled_attention_blocks > 0:
                        # print("=============== TEST =================")
                        # print("self_attn in ", name)
                        module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                        # module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else 'o_proj' if 'o_proj' in name else None
                        # print("module_name in ", module_name)
                        if module_name is not None:
                            grad = safe_get_full_grad(
                                param)  # (hidden_dim, head_dim)
                            if (module_name, layer_number
                                ) not in attention_warmup_grads:
                                attention_warmup_grads[(
                                    module_name,
                                    layer_number)] = grad.detach().cpu().to(
                                        torch.float32)
                                # attention_warmup_grads[(module_name, layer_number)] = grad.detach().to(torch.float32)

                            else:
                                attention_warmup_grads[(
                                    module_name,
                                    layer_number)] += grad.detach().cpu().to(
                                        torch.float32)
                                # attention_warmup_grads[(module_name, layer_number)] += grad.detach().to(torch.float32)

                            del grad

            # free GPU memory from training data/ memory
            del batch
            del outputs
            model.step()

            current_step_count += 1

            # time record ends, smt transfer and gradient select not included in time since they only happens in warm up iterations
            # print throughput every 200 steps
            if torch.distributed.get_rank() == 0 and step % 200 == 0:
                end = time.time()
                iter_time = end - start
                print_throughput(model.model, args, iter_time,
                                 args.global_rank)
            mean_loss += loss.item()

            training_loss_list.append(loss.item())

            if current_step_count % 100 == 0:
                print_rank_0(f"current_step_count: {current_step_count}")
                print_rank_0(
                    f"training_loss_list: {training_loss_list}",
                    args.global_rank)
                print_rank_0(
                    f"eval_loss_list: {eval_loss_list}",
                    args.global_rank)

            if current_step_count % args.eval_step == 0:
                # save best model, will lead to extra gpu memory usage
                ppl, eval_loss = evaluation(model, eval_dataloader, device)
                eval_loss_list.append(eval_loss)
                ppl_list.append(ppl)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if args.global_rank == 0:
                        # deep copy will lead to larger memory cost
                        best_model = copy.deepcopy(model.module).cpu()

                        # save best model, will lead to extra gpu memory usage
                        # print_rank_0(f"no save", args.global_rank)
                        # best_model_copy_state_dict = model.state_dict()#.to('cpu')

                    final_saved_model_index = current_step_count
                print_rank_0(
                    f"eval_loss_list: {eval_loss_list}",
                    args.global_rank)
                print_rank_0(
                    f"Validation perplexity: {ppl}, Validation loss: {eval_loss}",
                    args.global_rank)
            
            if current_step_count % 500 == 0:
                print_rank_0(
                    f"Saving model at step number {current_step_count}",
                    args.global_rank)
                iteration_save_model(model, eval_dataloader, device,
                                     best_eval_loss, current_step_count,
                                     tokenizer, args)

            if args.early_terminate and current_step_count % 100 == 0:
                # save best model, will lead to extra gpu memory usage
                iteration_save_model(model, eval_dataloader, device,
                                     best_eval_loss, current_step_count,
                                     tokenizer, args)
                if current_step_count % 3000 == 0:
                    final_eval_save_model(model, eval_dataloader, device,
                                          best_eval_loss,
                                          final_saved_model_index, tokenizer,
                                          best_model, args)
                    print_rank_0("epoch", epoch)
                    print_rank_0(f"training_loss_list: {training_loss_list}",
                                 args.global_rank)
                    print_rank_0(f"eval_loss_list: {eval_loss_list}",
                                 args.global_rank)
                    print_rank_0(f"ppl_list: {ppl_list}", args.global_rank)
                    exit()

        # print message and update_epoch_count function
        # source code is from: DeepspeedExamples:
        # https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py#L424
        print_rank_0(
            f"epoch {epoch + 1}/{args.num_ft_epochs} with training loss: {mean_loss / len(train_dataloader)}",
            args.global_rank)
        print_rank_0("epoch", epoch)
        print_rank_0(f"training_loss_list: {training_loss_list}",
                     args.global_rank)
        print_rank_0(f"eval_loss_list: {eval_loss_list}", args.global_rank)
        print_rank_0(f"ppl_list: {ppl_list}", args.global_rank)

        epoch_save_model(model, eval_dataloader, device, best_eval_loss,
                         epoch + 1, tokenizer, args)
        model.tput_timer.update_epoch_count()

    # final evaluation step after fine-tuning then save model
    final_eval_save_model(model, eval_dataloader, device, best_eval_loss,
                          final_saved_model_index, tokenizer, best_model, args)


if __name__ == "__main__":
    # parse_args source code example refer to DeepSpeed Example official website:
    # https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py#L308
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path',
                            action='append',
                            type=str,
                            help='The path to your fine-tuning dataset',
                            required=True)

        parser.add_argument(
            "--model_name_or_path",
            type=str,
            help=
            "Path to pretrained model or model identifier from huggingface.co/models.",
            required=True,
        )
        parser.add_argument(
            "--per_device_ft_batch_size",
            type=int,
            default=16,
            help="BS for ft process",
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=16,
            help="BS for evaluation process",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=2048,
            help="The maximum sequence length.",
        )
        parser.add_argument(
            "--eval_set_ratio",
            type=float,
            default=0.2, # 20% by default.
            help="Ratio of the dataset to use as validation.",
        )
        parser.add_argument(
            "--eval_step",
            type=int,
            default=30,
            help="interval steps between evaluation",
        )
        parser.add_argument(
            "--ft_learning_rate",
            type=float,
            default=9.65e-6,
            help="learning rate for full fine-tuning",
        )
        parser.add_argument("--w_decay",
                            type=float,
                            default=0.,
                            help="Weight decay")
        parser.add_argument("--num_ft_epochs",
                            type=int,
                            default=3,
                            help="Number of fine-tuning epochs")
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of minibatches before backward and optimizer update.",
        )
        parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default="linear",
            help="learning rate scheduler type",
            choices=["linear", "cosine"],
        )
        parser.add_argument(
            "--lr_warmup_steps",
            type=int,
            default=0,
            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--smt_lr_warmup_steps",
                            type=int,
                            default=0,
                            help="lr warmup after enter smt fine-tuning")
        parser.add_argument("--full_ft_steps",
                            type=int,
                            default=float('inf'),
                            help="full fine-tuning steps before SMT")
        parser.add_argument('--dtype',
                            type=str,
                            default='bf16',
                            choices=['fp16', 'bf16', 'fp32'],
                            help='fine-tuning data type')
        parser.add_argument("--output_dir",
                            type=str,
                            default=None,
                            help="Where to store the model.")
        parser.add_argument("--seed",
                            type=int,
                            default=1234,
                            help="Random seed")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="GPU id")

        parser.add_argument('--dropout',
                            type=float,
                            default=0.0,
                            help='set the dropout rate of the model')
        # deepspeed features
        parser.add_argument('--offload',
                            action='store_true',
                            help='zero offload or not')
        parser.add_argument(
            '--zero_stage',
            type=int,
            default=0,
            help='ZeRO optimization stage for Actor model (and clones).')

        ## Tokenizer
        parser.add_argument(
            "--add_eot_token",
            action='store_true',
            help="Add <|endoftext|> as additional special token to tokenizer")

        ## Hector TODO: low precision, recommend on step2, may test in the future condense NN project.
        parser.add_argument(
            '--compute_fp32_loss',
            action='store_true',
            help='Relevant for low precision dtypes (fp16, bf16, etc.). '
            'If specified, loss is calculated in fp32.')

        # SMT
        parser.add_argument('--matrix_sparsity',
                            action='store_true',
                            help='use SMT or not.')
        # deepspeed features
        parser.add_argument('--qk_scheduler',
                            action='store_true',
                            help='qk_different scheduler')

        # deepspeed features
        parser.add_argument(
            '--qk_lr_times',
            type=int,
            default=2,
            help=
            "Number of submatrix selected for mlp layers submatrix sparsity",
        )

        # deepspeed features
        parser.add_argument('--early_terminate',
                            action='store_true',
                            help='terminate in 200th iteration')
        parser.add_argument('--downsample_attention_blocks_ratio',
                            type=float,
                            default=0.0084,
                            help='Proportion of selected attention blocks relative to the total number of all blocks (not just attention blocks). Set to negative to turn it off.')
        parser.add_argument('--downsample_mlp_blocks_ratio',
                            type=float,
                            default=-1.0,
                            help='Proportion of selected mlp blocks relative to the total number of all blocks (not just mlp blocks).Set to negative to turn it off.')

        parser.add_argument(
            '--channel_sparsity',
            action='store_true',
            help=
            'use SMT, channel sparsity selected by massive activation or not.')

        parser.add_argument(
            "--num_mlp_channel",
            type=int,
            default=30,
            help=
            "Number of channel selected for mlp layers, activation based channel sparsity",
        )
        parser.add_argument(
            "--num_attention_channel",
            type=int,
            default=30,
            help=
            "Number of channels selected for attention layers, activation based channel sparsity",
        )

        parser.add_argument('--selection_strategy',
                            type=str,
                            default="no_restriction")

        parser.add_argument('--calculate_strategy',
                            type=str,
                            default="mean_abs")
        parser.add_argument(
            '--no_limit_mixture',
            action='store_true',
            help='pick blocks according to value without layers limitation')
        parser.add_argument(
            '--do_gradient_distribution_analysis',
            action='store_true',
            help='Perform gradient distribution analysis')
        parser.add_argument(
            "--smt_lr",
            type=float,
            default=5e-5,
            help=
            "Initial LoRA learning rate (after the potential warmup period) to use."
        )

        parser = deepspeed.add_config_arguments(parser)
        args = parser.parse_args()

        return args

    args = parse_args()
    trainer(args)

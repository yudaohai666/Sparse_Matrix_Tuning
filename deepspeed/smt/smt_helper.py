import torch
import numpy as np
from collections import defaultdict
import heapq
import re
from helpers.deepspeed_helpers import print_rank_0
import deepspeed
import matplotlib.pyplot as plt
import os

#
global_rank = torch.distributed.get_rank()

def analyze_gradient_distribution(gradients_per_key, key_string, output_dir):
    n_keys = len(gradients_per_key)
    n_cols = 3  # Adjust based on your preference
    n_rows = (n_keys + n_cols - 1) // n_cols


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_keys > 1 else [axes]  # Handle single-key case

    for ax, (key, values) in zip(axes, gradients_per_key.items()):
        arr = np.array(values)
        ax.hist(arr, bins=150, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Gradient Magnitude', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{key}')
        ax.legend()

    # Hide unused axes
    for i in range(n_keys, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    file_path = os.path.join(output_dir, f'gradient_histograms_{key_string}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

def select_submatrix_based_on_grads(grads,
                                    targeted_module_dims,
                                    n=660,
                                    selection_strategy="no_restriction",
                                    calculate_strategy="mean_abs",
                                    model="yahma/llama-13b-hf",
                                    do_gradient_distribution_analysis=False,
                                    output_dir=""):
    """
    grad: grad information for each MLP Linear weight matrix
    n: number of sub-matrix to choose
    """
    Block_dimension = 256

    block_means = {}
    for key, grad in grads.items():
        targeted_module_name = key[0]
        targeted_module_dim1 = int(targeted_module_dims[targeted_module_name][0] / Block_dimension)
        targeted_module_dim2 = int(targeted_module_dims[targeted_module_name][1] / Block_dimension)

        # Reshape the grad tensor into 256x256 blocks
        # if key[0] == 'gate_proj' or key[0] == 'up_proj' or key[0] == 'down_proj':
        #     # print(key[0], grad.size())
        #     print_rank_0(
        #         f"gate_proj / up_proj / down_proj dimension check:{key[0]}, {grad.size()}",
        #         global_rank)

        reshaped_grad = grad.reshape(targeted_module_dim1, Block_dimension, targeted_module_dim2,
                                    Block_dimension)

    # print("tensor shape:", reshaped_grad.shape)
        if calculate_strategy == 'mean_abs':
            block_means[key] = mean_abs(reshaped_grad)
        elif calculate_strategy == 'abs_mean':
            block_means[key] = abs_mean_(reshaped_grad)
        elif calculate_strategy == 'L1':
            block_means[key] = L1_norm(reshaped_grad)
        elif calculate_strategy == 'L2':
            block_means[key] = L2_norm(reshaped_grad)

    # for each linear layer, select certain number of sub-matrix, normal distributed selection
    if selection_strategy == "norm_dist":
        # Step 2: Rank all the blocks in all grad tensors using the abs.mean() value
        ranked_blocks = defaultdict(list)

        for key, block_mean in block_means.items():
            indices = torch.argsort(block_mean.view(-1), descending=True)
            # print("===================================================")
            # print("indices", indices)
            top_indices = indices[:n]
            for idx in top_indices:
                # may need to consider int memory cost in the future
                row = idx // block_mean.shape[1]
                col = idx % block_mean.shape[1]
                ranked_blocks[key].append((row.item(), col.item()))
        del indices
        del top_indices
        del key
        del block_mean
        # Step 3: Return the selected blocks and their corresponding information
        return ranked_blocks

    else:
        # Step 2: Use a min-heap to maintain top n blocks efficiently
        top_blocks = []

        gradients_per_key = {}
        for key in block_means.keys():
            gradients_per_key[key[0]] = [] 
        key_string = "_".join(str(k) for k in gradients_per_key.keys())

        for key, block_mean in block_means.items():
            for i in range(block_mean.shape[0]):
                for j in range(block_mean.shape[1]):
                    abs_mean = block_mean[i, j].item()
                    gradients_per_key[key[0]].append(abs_mean)
                    if len(top_blocks) < n:
                        heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                    else:
                        heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

        if do_gradient_distribution_analysis:
            analyze_gradient_distribution(gradients_per_key, key_string, output_dir)
        

        # print("===================================================")
        # print("top_blocks", top_blocks)

        # Step 3: Return the selected top n blocks and their corresponding information
        top_blocks.sort(
            reverse=True)  # Sort the top_blocks in descending order
        ranked_blocks = defaultdict(list)

        # selected_blocks = [(info, row, col, mean) for mean, (info, row, col) in top_blocks]

        # print("===================================================")
        # print("selected_blocks", selected_blocks)
        # for (info, row, col, mean) in selected_blocks:
        for mean, (info, row, col) in top_blocks:
            ranked_blocks[info].append((row, col))

        del top_blocks
        del mean
        del info
        del key
        del block_mean
        return ranked_blocks


def select_channel_based_on_activation(activation,
                                       n=660,
                                       selection_strategy="no_restriction",
                                       calculate_strategy="mean_abs",
                                       model="yahma/llama-13b-hf"):
    """
    # function need to revise: 1. select_submatrix_based_on_grads -> select_channel_based_on_activation
    # 2. freeze_unselected_matrix_layer -> freeze_unselected_channel_layer
    # 3. convert_linear_layer_to_matrix_sparsity -> convert_linear_layer_to_channel_sparsity

    activation: activation information for each MLP Linear layer input
    n: number of columns (dimension 1) in the activation tensor to choose
    selection_strategy: strategy to select channels (e.g., 'no_restriction', 'norm_dist')
    calculate_strategy: strategy to calculate channel importance ('mean_abs', 'L1', 'L2')
    model: model type to configure the function's behavior
    """

    # Step 1: Calculate metrics for each column in the activation tensors
    column_means = {}

    for key, act in activation.items():
        act = torch.sum(act.abs(), dim=0)
        # act is assumed to be a 2D tensor
        if calculate_strategy == 'mean_abs':
            column_means[key] = torch.mean(
                act.abs(), dim=0)  # Mean of absolute values along dimension 0
        elif calculate_strategy == 'abs_mean':
            column_means[key] = torch.abs(torch.mean(act,
                                                     dim=0))  # Absolute mean
        elif calculate_strategy == 'L1':
            column_means[key] = torch.norm(act, p=1,
                                           dim=0)  # L1 norm along dim=0
        elif calculate_strategy == 'L2':
            column_means[key] = torch.norm(act, p=2,
                                           dim=0)  # L2 norm along dim=0

    # Step 2: Select top columns based on the selection strategy
    if selection_strategy == "norm_dist":
        # Rank the columns by the calculated metric value
        ranked_columns = defaultdict(list)

        for key, column_mean in column_means.items():
            indices = torch.argsort(
                column_mean,
                descending=True)  # Sort column-wise values in descending order
            top_indices = indices[:n]  # Get top n column indices
            ranked_columns[key] = top_indices.tolist(
            )  # Store column indices as a list

        del indices
        del top_indices
        del key
        del column_mean

        return ranked_columns

    else:
        # Use a min-heap to efficiently select top n columns
        top_columns = []
        for key, column_mean in column_means.items():
            for idx in range(column_mean.shape[0]):
                value = column_mean[idx].item()
                if len(top_columns) < n:
                    heapq.heappush(top_columns, (value, (key, idx)))
                else:
                    heapq.heappushpop(top_columns, (value, (key, idx)))

        top_columns.sort(
            reverse=True)  # Sort the top columns in descending order by value

        ranked_columns = defaultdict(list)

        # Fill the result dictionary with selected column indices
        for value, (key, idx) in top_columns:
            ranked_columns[key].append(idx)

        del top_columns
        del value
        del key
        del column_mean

        return ranked_columns


def mean_abs(grad_tensor):
    # print_rank_0(f"use mean()abs() as calculation strategy", global_rank)
    return grad_tensor.mean(dim=(1, 3)).abs()


def abs_mean_(grad_tensor):
    # print_rank_0(f"use abs()mean() as calculation strategy", global_rank)
    return grad_tensor.abs().mean(dim=(1, 3))


def L1_norm(grad_tensor):
    # print_rank_0(f"use L1 norm as calculation strategy", global_rank)

    return grad_tensor.abs().sum(dim=(1, 3))


def L2_norm(grad_tensor):
    # print_rank_0(f"use L2 norm as calculation strategy", global_rank)
    return torch.sqrt(torch.sum(grad_tensor.abs()**2, dim=(1, 3)))


def replacement_test(ref_param, model_param, x_index, y_index):
    matrix_equal = torch.equal(ref_param, model_param)
    if matrix_equal:

        print("The ref_param and param have the same values.")
    else:
        print("The ref_param and param have different values.")
    print("====================================================")

    matrix_equal1 = torch.equal(
        ref_param[x_index:x_index + 256, y_index:y_index + 256],
        model_param[x_index:x_index + 256, y_index:y_index + 256])
    if matrix_equal1:
        print("The selected region have the same values.")
    else:
        print("The selected region have different values.")


def get_blocks(model):
    from transformers.models.bloom.modeling_bloom import BloomForCausalLM
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def get_named_linears(module):
    import torch.nn as nn
    return {
        name: m
        for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }


############################################# Test Code ####################################

if __name__ == '__main__':

    # # Example 1 usage:
    # grads = {
    #     ('gate_proj', 1): torch.zeros(11008, 4096),
    #     ('up_proj', 1): torch.zeros(11008, 4096),
    #     ('down_proj', 2): torch.ones(4096, 11008)
    # }
    #
    # grads[('gate_proj', 1)][0:256, 0:256] = torch.ones(256, 256)*10
    # grads[('gate_proj', 1)][256:512, 0:256] = torch.ones(256, 256)*10
    # grads[('up_proj', 1)][0:2560, 0:256] = torch.ones(2560, 256)*10
    # selected_blocks = select_submatrix_based_on_grads(grads, n=20, selection_strategy = "no_restriction")
    # print(selected_blocks)
    # print(len(selected_blocks))

    # # Example 1 usage:
    activation = {
        ('gate_proj', 1): torch.zeros(3, 11008, 4096),
        ('up_proj', 1): torch.zeros(3, 11008, 4096),
        ('down_proj', 2): torch.ones(3, 4096, 11008)
    }

    activation[('gate_proj', 1)][:, :, 0:256] = torch.ones(3, 11008, 256) * 1
    activation[('gate_proj', 1)][:, :, 0:4] = torch.ones(3, 11008, 4) * 10
    activation[('up_proj', 1)][:, :, 3:6] = torch.ones(3, 11008, 3) * 100
    activation[('down_proj', 2)][:, :, 3:6] = torch.ones(3, 4096, 3) * 100

    selected_channel = select_channel_based_on_activation(
        activation, n=100, selection_strategy="no_restriction")
    print(selected_channel)
    print(len(selected_channel))

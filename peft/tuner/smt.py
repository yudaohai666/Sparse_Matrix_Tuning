import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from collections import defaultdict
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


Block_dimension = 256





class SMTModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted
        config ([`SMTConfig`]): The configuration of the SMT model.

    Returns:
        `torch.nn.Module`: The SMT model.

    Example::

        from transformers import AutoModelForSeq2SeqLM, LoraConfig
        from peft import LoraModel, LoraConfig
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        attention_grads = {}
        mlp_grads = {}
        lora_model = LoraModel(config, model, attention_grads, mlp_grads)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SMTConfig`]): The configuration of the SMT model.
        - **attention_grads** (['dict']): dictionary of attention gradient information.
        - **mlp_grads** (['dict']):  dictionary of mlp gradient information.
    """

    def __init__(self, config, model, attention_grads, mlp_grads):
        super().__init__()
        self.peft_config = config
        self.model = model
        selected_submatrix_mlp = {}
        selected_submatrix_attention = {}
        if self.peft_config.num_submatrix_mlp > 0:
            selected_submatrix_mlp = select_submatrix_based_on_grads(mlp_grads,
                                                                 self.peft_config.num_submatrix_mlp,
                                                                 selection_strategy=self.peft_config.selection_strategy,
                                                                 calculate_strategy=self.peft_config.calculate_strategy,
                                                                 model=self.peft_config.model_name_or_path)
        if self.peft_config.num_submatrix_attn > 0:
            selected_submatrix_attention = select_submatrix_based_on_grads(attention_grads,
                                                                           self.peft_config.num_submatrix_attn,
                                                                           selection_strategy=self.peft_config.selection_strategy,
                                                                           calculate_strategy=self.peft_config.calculation_strategy,
                                                                           model=self.peft_config.model_name_or_path)
        self.model = mark_only_smt_as_trainable(self.model, selected_submatrix_mlp, selected_submatrix_attention)  
        # self.model = mark_only_smt_as_trainable(self.model.module, selected_submatrix_mlp, selected_submatrix_attention)
        self.model = self.convert_linear_layer_to_matrix_sparsity(self.model, selected_submatrix_mlp, selected_submatrix_attention)
        self.print_trainable_parameters()
        # 8-bit quantization function not implemented yet
        # self._find_and_replace()
        self.forward = self.model.forward

    def convert_linear_layer_to_matrix_sparsity(self, model, selected_submatrix, selected_submatrix_attention,
                                                part_module_name=['.layers']):
        pattern = re.compile(r'model\.layers\.(\d+)\.')

        replace_name = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(part in name for part in part_module_name):
                # print(f"convert {name} to LoRA")
                replace_name.append(name)
        for name in replace_name:
            if "mlp" in name:
                module = recursive_getattr(model, name)
                if module.weight.requires_grad:
                    module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                    match = pattern.search(name)
                    layer_number = int(match.group(1)) if match else None

                    # index_list: list of index which require_grad, need to pass into Linear
                    index_list = selected_submatrix[(module_name, layer_number)]

                    tmp = LinearLayer_MatrixSparsity(
                        module.weight,
                        bias=None,
                        index_list=index_list).to(module.weight.device).to(module.weight.dtype)
                    recursive_setattr(model, name, tmp)
            if "self_attn" in name:
                module = recursive_getattr(model, name)
                if module.weight.requires_grad:
                    module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                    match = pattern.search(name)
                    layer_number = int(match.group(1)) if match else None

                    # index_list: list of index which require_grad, need to pass into Linear
                    index_list = selected_submatrix_attention[(module_name, layer_number)]

                    tmp = LinearLayer_MatrixSparsity(
                        module.weight,
                        bias=None,
                        index_list=index_list).to(module.weight.device).to(module.weight.dtype)
                    recursive_setattr(model, name, tmp)

        return model

    # def _find_and_replace(self):
    #     loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
    #     if loaded_in_8bit and not is_bnb_available():
    #         raise ImportError(
    #             "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
    #             "You can install it with `pip install bitsandbytes`."
    #         )
    #     is_target_modules_in_base_model = False
    #     is_hf_device_map_available = hasattr(self.model, "hf_device_map")
    #     kwargs = {
    #         "r": self.peft_config.r,
    #         "lora_alpha": self.peft_config.lora_alpha,
    #         "lora_dropout": self.peft_config.lora_dropout,
    #         "fan_in_fan_out": self.peft_config.fan_in_fan_out,
    #         "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
    #                          and not is_hf_device_map_available,
    #         "dora_simple": self.peft_config.dora_simple
    #     }
    #     key_list = [key for key, _ in self.model.named_modules()]
    #     for key in key_list:
    #         if isinstance(self.peft_config.target_modules, str):
    #             target_module_found = re.fullmatch(self.peft_config.target_modules, key)
    #         else:
    #             target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
    #
    #         if isinstance(self.peft_config.Wdecompose_target_modules, str):
    #             wdecompose_target_module_found = re.fullmatch(self.peft_config.Wdecompose_target_modules, key)
    #         elif self.peft_config.Wdecompose_target_modules == None:
    #             wdecompose_target_module_found = False
    #         else:
    #             wdecompose_target_module_found = any(
    #                 key.endswith(target_key) for target_key in self.peft_config.Wdecompose_target_modules)
    #
    #         if target_module_found:
    #             if not is_target_modules_in_base_model:
    #                 is_target_modules_in_base_model = True
    #             parent, target, target_name = self._get_submodules(key)
    #             bias = target.bias is not None
    #             if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
    #                 kwargs.update(
    #                     {
    #                         "has_fp16_weights": target.state.has_fp16_weights,
    #                         "memory_efficient_backward": target.state.memory_efficient_backward,
    #                         "threshold": target.state.threshold,
    #                         "index": target.index,
    #                     }
    #                 )
    #                 if self.peft_config.enable_lora is None:
    #                     new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
    #                 else:
    #                     raise NotImplementedError
    #
    #             elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
    #                 new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
    #             elif self.peft_config.enable_lora is not None:
    #                 raise NotImplementedError
    #
    #             self._replace_module(parent, target_name, new_module, target)
    #
    #         elif wdecompose_target_module_found:
    #             if not is_target_modules_in_base_model:
    #                 is_target_modules_in_base_model = True
    #             parent, target, target_name = self._get_submodules(key)
    #             bias = target.bias is not None
    #             if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
    #                 kwargs.update(
    #                     {
    #                         "has_fp16_weights": target.state.has_fp16_weights,
    #                         "memory_efficient_backward": target.state.memory_efficient_backward,
    #                         "threshold": target.state.threshold,
    #                         "index": target.index,
    #                     }
    #                 )
    #                 if self.peft_config.enable_lora is None:
    #                     new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
    #                 else:
    #                     raise NotImplementedError
    #
    #             elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
    #                 new_module = Linear(target.in_features, target.out_features, bias=bias, Wdecompose=True, **kwargs)
    #             elif self.peft_config.enable_lora is not None:
    #                 raise NotImplementedError
    #             self._replace_module(parent, target_name, new_module, target)
    #
    #     if not is_target_modules_in_base_model:
    #         raise ValueError(
    #             f"Target modules {self.peft_config.target_modules} not found in the base model. "
    #             f"Please check the target modules and try again."
    #         )

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        # for param in self.model.module.parameters():
        for param in self.model.parameters():
        # for param in self.model.parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        #
        with torch.no_grad():
            magnitude = (torch.linalg.norm(new_module.weight.detach(), dim=1)).unsqueeze(1).detach()
            new_module.weight_m_wdecomp.weight.copy_(magnitude)

        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "weight_m_wdecomp" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config
    #
    # def _set_adapter_layers(self, enabled=True):
    #     for module in self.model.modules():
    #         if isinstance(module, LoraLayer):
    #             module.disable_adapters = False if enabled else True
    #
    # def enable_adapter_layers(self):
    #     self._set_adapter_layers(enabled=True)
    #
    # def disable_adapter_layers(self):
    #     self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

def mark_only_smt_as_trainable(model, select_parameters, select_attention_parameters, mixture = False):
    # selected_parameters: (module_name, layer_number, head_number)
    # model = convert_selected_sau_to_linear_layer(model, select_parameters, exclude)
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in model.named_parameters():
        if mixture:
            if "mlp" in name:
                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_parameters.keys():
                    param.requires_grad = True

                    # print("selected grad True layer")
                    # print(module_name, layer_number)
                else:
                    param.requires_grad = False
                    print(f"Layer set to grad = Flase:{name}")

            elif "self_attn" in name:
                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_parameters.keys():
                    param.requires_grad = True
                    print(f"Layer set to grad = True:{name}")


                else:
                    param.requires_grad = False
                    print(f"Layer set to grad = Flase:{name}")

            else:
                param.requires_grad = False
                print(f"Layer set to grad = False:{name}")

        else:
            if "mlp" in name:
                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_parameters.keys():
                    param.requires_grad = True
                    print(f"Layer set to grad = True:{name}")

                    # print("selected grad True layer")
                    # print(module_name, layer_number)
                else:
                    param.requires_grad = False
                    print(f"Layer set to grad = Flase:{name}")

            elif "self_attn" in name:
                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_attention_parameters.keys():
                    param.requires_grad = True
                    print(f"Layer set to grad = True:{name}")


                else:
                    param.requires_grad = False
                    print(f"Layer set to grad = Flase:{name}")

            else:
                param.requires_grad = False
                print(f"Layer set to grad = False:{name}")

    return model



class LinearLayer_MatrixSparsity(torch.nn.Module):
    # an simple implementation of matrix sparsity
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 bias=None,
                 index_list = []):
        super(LinearLayer_MatrixSparsity, self).__init__()
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        self.index_list = index_list


        self.selected_weight = torch.empty(len(index_list) * Block_dimension, Block_dimension,dtype=self.weight.data.dtype,
                                  device=self.weight.data.device)

        for i in range(len(index_list)):
            index = index_list[i]
            self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]
        self.selected_weight.requires_grad = True
        self.selected_weight = nn.Parameter(self.selected_weight)


        self.fn = linearZ.apply

    def forward(self, x):
        for i in range(len(self.index_list)):
            index = self.index_list[i]
            # self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]
            self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension] = self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :]

        x = self.fn(x,  self.selected_weight, self.index_list, self.weight)
        return x

class linearZ(torch.autograd.Function):
    # only support batch size D=3 now, for batch size = 1, need to add mm. operation.
    @staticmethod
    def forward(ctx, input, selected_weight, matrix_index_list, weight):
        input_list = []
        for index in matrix_index_list:
            input_list.append(input[:, :, index[1]*Block_dimension: index[1]*Block_dimension+Block_dimension])
        # save for backward may only support tensor, use others to save!
        ctx.list1 = input_list
        ctx.list2 = matrix_index_list

        ctx.save_for_backward(weight)


        # output = input.mm(weight.t())
        # print("input size:",input.size())
        # print("weight size:",weight.data.size())
        output = torch.matmul(input, weight.t())


        # memory free
        del weight
        del input_list
        del matrix_index_list


        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight,  = ctx.saved_tensors
        input_list = ctx.list1
        matrix_index_list = ctx.list2

        # Pytorch use C++ engine to check whether gradient has matched dimenstion or not
        grad_weight = torch.empty(len(input_list) * Block_dimension, Block_dimension,dtype=grad_output.dtype,
                                  device=grad_output.device)
        for i in range(len(input_list)):
            index = matrix_index_list[i]

            # print("index:", index)
            # print("grad_output_dim:", grad_output.size())
            # tmp = grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :]
            # print("tmp size", tmp.size())
            # print("input list[i]", input_list[i].size())
            # tmp1 = torch.matmul(tmp, input_list[i])
            # grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(tmp1, dim=0)

            grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(torch.matmul(grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :], input_list[i]), dim=0)

        grad_input = torch.matmul(grad_output, weight)

        # memory free
        del weight
        del input_list
        del matrix_index_list

        return grad_input, grad_weight, None, None




def select_submatrix_based_on_grads(grads, n=660, selection_strategy = "no_restriction", calculate_strategy = "mean_abs", model = "yahma/llama-13b-hf"):
    """
    grad: grad information for each MLP Linear weight matrix
    n: number of sub-matrix to choose
    """
    # Step 1: Calculate absolute value of mean for all grad tensors in every 256x256 block
    if (model == "yahma/llama-13b-hf") or (model == "NousResearch/Llama-2-13b-hf"):
        Block_dimension = 256
        large_d = 54
        small_d = 20
    # elif model == "yahma/llama-7b-hf":
    elif (model == "NousResearch/Llama-2-7b-hf") or (model == "meta-llama/Llama-2-7b-hf") or (model == "yahma/llama-7b-hf") or (model == "meta-llama/Llama-2-7b-chat-hf"):
        Block_dimension = 256
        large_d = 43
        small_d = 16
    elif (model == "NousResearch/Meta-Llama-3-8B") or (model == "meta-llama/Meta-Llama-3-8B"):
        Block_dimension = 256
        large_d = 56
        small_d = 16

    block_means = {}
    for key, grad in grads.items():
        # Reshape the grad tensor into 256x256 blocks
        if key[0] == 'gate_proj' or key[0] == 'up_proj':
            # print(key[0], grad.size())
            print(f"gate_proj and up_proj dimension check:{key[0]}, {grad.size()}")

            reshaped_grad = grad.reshape(large_d, Block_dimension, small_d, Block_dimension)

        elif key[0] == 'down_proj':
            # print(key[0], grad.size())
            print(f"down_proj dimension check:{key[0]}, {grad.size()}")

            reshaped_grad = grad.reshape(small_d, Block_dimension, large_d, Block_dimension)


        elif key[0] == 'q_proj' or key[0] == 'k_proj' or key[0] == 'v_proj':
            print(f"qkv dimension check:{key[0]}, {grad.size()}")
            if (model == "meta-llama/Meta-Llama-3-8B") and (key[0] == 'k_proj' or key[0] == 'v_proj'):
                small_d_ = 4
                reshaped_grad = grad.reshape(small_d_, Block_dimension, small_d, Block_dimension)
            else:
                reshaped_grad = grad.reshape(small_d, Block_dimension, small_d, Block_dimension)

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
        for key, block_mean in block_means.items():
            for i in range(block_mean.shape[0]):
                for j in range(block_mean.shape[1]):
                    abs_mean = block_mean[i, j].item()
                    if len(top_blocks) < n:
                        heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                    else:
                        heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

        # print("===================================================")
        # print("top_blocks", top_blocks)

        # Step 3: Return the selected top n blocks and their corresponding information
        top_blocks.sort(reverse=True)  # Sort the top_blocks in descending order
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


def mean_abs(grad_tensor):
    print(f"use mean()abs() as calculation strategy")
    return grad_tensor.mean(dim=(1, 3)).abs()

def abs_mean_(grad_tensor):
    print(f"use abs()mean() as calculation strategy")
    return grad_tensor.abs().mean(dim=(1, 3))

def L1_norm(grad_tensor):
    print(f"use L1 norm as calculation strategy")

    return grad_tensor.abs().sum(dim=(1, 3))

def L2_norm(grad_tensor):
    print(f"use L2 norm as calculation strategy")
    return torch.sqrt(torch.sum(grad_tensor.abs() ** 2, dim=(1, 3)))





# Source Code from DeepSpeed Examples official website
# Please refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/utils.py#L210
def get_optimizer_sparse_grouped_parameters(
    model,
    weight_decay,
    smt_lr,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):


    print(f"================ PRINT PARAM NAME [0]=======================")

    for name, param in model.named_parameters():
        if (not any(nd in name.lower() for nd in no_decay_name_list)
                and param.requires_grad and not any(nd in name.lower() for nd in lora_name_list)):
            print(f"name0:{name}")

    print(f"================ PRINT PARAM NAME [1]=======================")
    for n, p in model.named_parameters():
        if (not any(nd in n.lower() for nd in no_decay_name_list)
                and p.requires_grad and any(nd in n.lower() for nd in lora_name_list)):
            print(f"name1:{n}")



    print(f"================ PRINT PARAM NAME [2]=======================")
    for n, p in model.named_parameters():
        if (any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad):
            print(f"name2:{n}")



    optimizer_grouped_parameters = [
        {
            "params": #tmp
            [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower() for nd in lora_name_list))
            ]
            ,
            "weight_decay":
            weight_decay,
            "lr":
            smt_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower() for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)

    print(f"group parameters: {non_empty_groups}")

    return non_empty_groups #, sorted_selected_submatrix




# class LoraLayer:
#     def __init__(
#             self,
#             r: int,
#             lora_alpha: int,
#             lora_dropout: float,
#             merge_weights: bool,
#     ):
#         self.r = r
#         self.lora_alpha = lora_alpha
#         # Optional dropout
#         if lora_dropout > 0.0:
#             self.lora_dropout = nn.Dropout(p=lora_dropout)
#         else:
#             self.lora_dropout = lambda x: x
#         # Mark the weight as unmerged
#         self.merged = False
#         self.merge_weights = merge_weights
#         self.disable_adapters = False
#
#
# class Linear(nn.Linear, LoraLayer):
#     # Lora implemented in a dense layer
#     def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             r: int = 0,
#             lora_alpha: int = 1,
#             lora_dropout: float = 0.0,
#             fan_in_fan_out: bool = False,
#             # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#             merge_weights: bool = True,
#             Wdecompose: bool = False,
#             dora_simple: bool = True,
#             **kwargs,
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
#
#         self.weight_m_wdecomp = nn.Linear(1, out_features,
#                                           bias=False)  # self.weight_m_wdecomp.weight # shape: out_features, 1
#
#         self.fan_in_fan_out = fan_in_fan_out
#         self.Wdecompose = Wdecompose  # whether to tune only the magnitude component of Wdecompose or not
#         self.dora_simple = dora_simple  # whether to use dora simple to save up GPU memory
#         if self.Wdecompose == False:
#             if r > 0:
#                 self.lora_A = nn.Linear(in_features, r, bias=False)
#                 self.lora_B = nn.Linear(r, out_features, bias=False)
#                 self.scaling = self.lora_alpha / self.r
#                 # Freezing the pre-trained weight matrix
#
#         self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.T
#
#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, "lora_A"):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B.weight)
#
#     def train(self, mode: bool = True):
#         nn.Linear.train(self, mode)
#         if self.Wdecompose == False:
#             self.lora_A.train(mode)
#             self.lora_B.train(mode)
#         self.weight_m_wdecomp.train(mode)
#
#         if not mode and self.merge_weights and not self.merged:
#             # Merge the weights and mark it
#             if self.Wdecompose:
#                 norm_scale = (self.weight_m_wdecomp.weight / (torch.linalg.norm(self.weight, dim=1)).unsqueeze(1))
#                 weight = norm_scale * self.weight
#                 self.weight.data.copy_(weight.detach())
#             else:
#                 if self.r > 0:
#                     new_weight_v = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight,
#                                                            fan_in_fan_out=self.fan_in_fan_out) * self.scaling
#                     weight = (self.weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v, dim=1)).unsqueeze(
#                         1)) * new_weight_v
#                     self.weight.data.copy_(weight.detach())
#             self.merged = True
#         elif self.merge_weights and self.merged:
#             raise NotImplementedError
#
#     def eval(self):
#         nn.Linear.eval(self)
#         if self.Wdecompose == False:
#             self.lora_A.eval()
#             self.lora_B.eval()
#         self.weight_m_wdecomp.eval()
#
#     def forward(self, x: torch.Tensor):
#         previous_dtype = self.weight.dtype
#
#         if self.disable_adapters:
#             raise NotImplementedError
#
#         elif self.Wdecompose and not self.merged:
#
#             norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight, dim=1))
#
#             org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
#
#             result = org_result + (norm_scale - 1) * (
#                 F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
#
#             if not self.bias is None:
#                 result += self.bias.view(1, -1).expand_as(result)
#
#         elif self.r > 0 and not self.merged:
#
#             new_weight_v = self.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
#
#             if self.dora_simple:
#                 norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()
#             else:
#                 norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v, dim=1))
#
#             org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
#
#             dropout_x = self.lora_dropout(x)
#
#             result = org_result + (norm_scale - 1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
#
#             if not self.bias is None:
#                 result += self.bias.view(1, -1).expand_as(result)
#
#             result += (norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
#
#         else:
#             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
#
#         if result.dtype != previous_dtype:
#             result = result.to(previous_dtype)
#
#         return result
#
#
# class MergedLinear(nn.Linear, LoraLayer):
#     # Lora implemented in a dense layer
#     def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             r: int = 0,
#             lora_alpha: int = 1,
#             lora_dropout: float = 0.0,
#             enable_lora: List[bool] = [False],
#             fan_in_fan_out: bool = False,
#             merge_weights: bool = True,
#             **kwargs,
#     ):
#         raise NotImplementedError


# if is_bnb_available():
#     class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
#         # Lora implemented in a dense layer
#         def __init__(
#                 self,
#                 in_features,
#                 out_features,
#                 r: int = 0,
#                 lora_alpha: int = 1,
#                 lora_dropout: float = 0.0,
#                 Wdecompose: bool = False,
#                 **kwargs,
#         ):
#             raise NotImplementedError
#
#
#     class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
#         # Lora implemented in a dense layer
#         def __init__(
#                 self,
#                 in_features: int,
#                 out_features: int,
#                 r: int = 0,
#                 lora_alpha: int = 1,
#                 lora_dropout: float = 0.0,
#                 enable_lora: List[bool] = [False],
#                 **kwargs,
#         ):
#             raise NotImplementedError
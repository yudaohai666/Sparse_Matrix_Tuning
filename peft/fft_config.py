import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    GenerationConfig,
)

import math
from typing import Optional, List, Union
import json
from dataclasses import dataclass, field
from config import PeftConfig, PeftType
# from peft import LoraConfig

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


@dataclass
class FFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].
    This is the configuration for full fine-tuning
    Args:

    """

    num_submatrix_mlp: int = field(default=0, metadata={"help": "number of submatrices applied to MLP layers"})
    num_submatrix_attn: int = field(default=0, metadata={"help": "number of submatrices applied to Attention layers"})
    smt_dropout: Optional[float] = field(default=0.0)
    # to fix, need to be more flexible
    model_name: str = field(
        default=None,
        metadata={
            "help" : "name of the llama model"
            "choose from ['yahma/llama-13b-hf', 'NousResearch/Llama-2-13b-hf', 'NousResearch/Llama-2-7b-hf', 'yahma/llama-7b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'NousResearch/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B']"
        },
    )
    full_ft_steps: int = field(default=100, metadata={"help": "number of iterations for full fine-tuning before SMT"})
    selection_strategy: str = field(default="no_restriction", metadata={"help": "sub-matrices distribution selection strategy"})
    calculation_strategy: str = field(default="mean_abs", metadata={"help": "gradient calculation within submatrices"})


    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v', 'k'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.SMT






def fft_create_and_prepare_model(args, data_args, training_args):


    peft_config = None
    peft_config = FFTConfig(
        num_submatrix_mlp = args.num_submatrix_mlp,
        num_submatrix_attn = args.num_submatrix_attn,
        smt_dropout = args.smt_dropout,
        model_name = args.model_name,
        full_ft_steps = args.full_ft_steps,
        selection_strategy = args.selection_strategy,
        calculation_strategy = args.calculation_strategy,
        target_modules = args.target_modules,
        merge_weights = False,
    )




    add_special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        add_special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        add_special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    # source code from: Deepspeed example website, load_hf_tokenizer
    # https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
    if os.path.exists(args.model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(args.model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             args.model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=True)
    else:
        tokenizer = get_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    tokenizer.model_max_length = args.max_seq_len

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            dropout=args.smt_dropout)




    return model, peft_config, tokenizer

# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if model_name_or_path == "meta-llama/Meta-Llama-3-8B":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)      # not adding start token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    else:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)       # not adding start token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'

    return tokenizer


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    dropout = None
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # actually i do not know why we need this, but commenting it causes cuda error
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]



def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data

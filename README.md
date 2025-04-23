<h3 align="center">
    <p>SMT: Fine-Tuning Large Language Models with Sparse Matrices </p>
</h3>

The implementation of [SMT: Fine-Tuning Large Language Models with Sparse Matrices](https://openreview.net/forum?id=GbgCRJedQ7). This paper introduce a method for selecting sparse sub-matrices that aims to minimize the performance gap between PEFT vs. full fine-tuning (FT) while also reducing both fine-tuning computational costs and memory costs. We explored both gradient-based and activation-based parameter selection methods to identify the most significant sub-matrices for downstream tasks, updating only these blocks during fine-tuning. In our experiments, we demonstrated that SMT consistently surpasses other PEFT baselines (e.g., LoRA and DoRA) in fine-tuning popular large language models such as LLaMA across a broad spectrum of tasks, while reducing the GPU memory footprint by 67% compared to FT. 


We implemented SMT in two frameworks: DeepSpeed and Hugging Face Trainer/PEFT. Instructions for setting up the environment, training, and evaluation can be found in the subfolders.


## Latest News 🔥🔥

* [2025-04-10] We can support Deepseek-R1-Distill model now!
* [2025-02-01] Our paper, [SMT: Fine-Tuning Large Language Models with Sparse Matrices](https://openreview.net/forum?id=GbgCRJedQ7), has been accepted by ICLR 2025.🎉🎉
* [2024-07-10] Our paper is on [ArXiv](https://arxiv.org/abs/2405.15525). 



## Latest Results on DeepSeek-R1-Distill Model 

>Obsrervation 1: Deepseek-R1-Distill-LLaMA8B model underperforms the base LLaMA-3-8B model on Commonsense Reasoning dataset without reasoning trace.

|DeepSeek-R1-Distill-LLaMA8B| BoolQ | PIQA  | SIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA  | AVG   |
|---------------------------|-------|-------|-------|-----------|------------|-------|-------|-------|-------|
| base                      | 53.9  | 50.0  | 37.4  | 23.4      | 25.3       | 30.6  | 28.0  | 23.4  | 34.0  |
| SMT(0.86%)                | 70.6  | 66.4  | 77.8  | 62.4      | 84.6       | 60.9  | 53.2  | 72.6  | 68.6  |
| Full FT.                  | 71.0  | 66.2  | 76.5  | 62.2      | 85.4       | 61.8  | 52.8  | 72.6  | 68.6  |


|LLaMA3-8B Model| BoolQ | PIQA  | SIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA  | AVG   |
|---------------------------|-------|-------|-------|-----------|------------|-------|-------|-------|-------|
| SMT(0.71%)    | 75.7  | 88.4  | 81.4  | 96.2      | 88.2       | 92.7  | 83.2  | 88.6  | 86.8  |


>Observation 2: Deepseek-R1-Distill-LLaMA8B model largely outperforms the base LLaMA-3-8B model on Math Reasoning dataset with reasoning trace.

|LLaMA3-8B Model| GSM8k | SingleEq  | SVAMP  | MultiArith | AddSub | AQuA | AVG   |
|---------------------------|-------|-----------|--------|------------|--------|----|----------|
| SMT(0.71%)    | 42.8  | 88.5  | 60.4  | 93.9      | 85.8       |  25.2  |  66.1|

|DeepSeek-R1-DistillLLaMA8B Model| GSM8k | SingleEq  | SVAMP  | MultiArith | AddSub | AQuA | AVG   |
|---------------------------|-------|-----------|--------|------------|--------|----|----------|
| SMT(0.71%)    | 60.8  | 92.5  | 70.6  | 95.3      | 87.3       |  31.2  |  73.0|


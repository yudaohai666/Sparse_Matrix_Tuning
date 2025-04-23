<h3 align="center">
    <p>SMT: Fine-Tuning Large Language Models with Sparse Matrices </p>
</h3>

The implementation of [SMT: Fine-Tuning Large Language Models with Sparse Matrices](https://openreview.net/forum?id=GbgCRJedQ7). This paper introduce a method for selecting sparse sub-matrices that aims to minimize the performance gap between PEFT vs. full fine-tuning (FT) while also reducing both fine-tuning computational costs and memory costs. We explored both gradient-based and activation-based parameter selection methods to identify the most significant sub-matrices for downstream tasks, updating only these blocks during fine-tuning. In our experiments, we demonstrated that SMT consistently surpasses other PEFT baselines (e.g., LoRA and DoRA) in fine-tuning popular large language models such as LLaMA across a broad spectrum of tasks, while reducing the GPU memory footprint by 67% compared to FT. 


We implemented SMT in two frameworks: DeepSpeed and Hugging Face Trainer/PEFT. Instructions for setting up the environment, training, and evaluation can be found in the subfolders.


## Latest News 🔥🔥

* [2025-04-10] We can support Deepseek-R1-Distill model now!
* [2025-02-01] Our paper, [SMT: Fine-Tuning Large Language Models with Sparse Matrices](https://openreview.net/forum?id=GbgCRJedQ7), has been accepted by ICLR 2025.:tada::tada:
* [2024-07-10] Our paper is on [ArXiv](https://arxiv.org/abs/2405.15525). 

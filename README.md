# Introduction

this project is to provide a tool to extract data item from patent materials

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## entity and relation extraction

```shell
python3 main.py --input_dir <path/to/directory/of/patents> [--output_dir <path/to/output/directory>] [--ckpt <path/to/customized/ckpt>]
```

# finetune

to improve the LLM on ability to extract electrolyte related information, we use supervised finetuning to moderate pretrain LLM's behavior.

## generate dataset

```shell
python3 create_dataset.py --input datasets/origin.json --output datasets/trainset.jsonl
```

## finetuning LLM

```shell
python3 finetune.py --pretrained_ckpt <hugging/face/model/id> --sft_ckpt <path/to/ckpt> --dataset <path/to/dataset> --device (cuda|cpu)
```

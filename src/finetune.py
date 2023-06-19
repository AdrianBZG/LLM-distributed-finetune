
import ray
import argparse
import torch
import os
import json
import pandas as pd
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from ray.data.preprocessors import BatchMapper
from ray.train.huggingface import TransformersTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from accelerate.utils import DummyOptim, DummyScheduler 
from peft import get_peft_model_state_dict
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from ray.air import session

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ['query_key_value']


def trainer_init_per_worker(train_dataset, eval_dataset=None, **config):
    print(f"Is CUDA available? {torch.cuda.is_available()}")

    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load tokenizer
    model_id = config.get('model')
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.use_cache = False

    # Enable gradient checkpointing and Quantitize
    #model.gradient_checkpointing_enable(). # Not supported by this model :(. See https://github.com/huggingface/accelerate/issues/389
    model = prepare_model_for_int8_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.get('lora').get('r'),
        lora_alpha=config.get('lora').get('alpha'),
        target_modules=config.get('lora').get('target_modules'),
        lora_dropout=config.get('lora').get('dropout'),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Training Configuration
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        #auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        save_total_limit=4,
        logging_steps=5,
        save_strategy='steps',
        weight_decay=0,
        push_to_hub=False,
        disable_tqdm=True,
        no_cuda=not config.get('platform').get('use_gpu'),
        gradient_checkpointing=True,
        output_dir="./outputs_ray",
        ddp_find_unused_parameters=False,
        deepspeed=config.get('platform').get('deepspeed')
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Configure the Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator
    )
    model.config.use_cache = False

    return trainer


def prepare_dataset(path, model_id):
    dataset = load_dataset("json", data_files=path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_prompts = {}
    dataset_prompts['text'] = []

    def generate_prompt(data_point):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately 
    completes the request.  # noqa: E501
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    ### Response:
    {data_point["output"]}"""

    for data_point in dataset:
        prompt = generate_prompt(data_point)
        dataset_prompts['text'].append(prompt)

    # Transform to Ray dataset format
    dataset_prompts_df = pd.DataFrame.from_dict(dataset_prompts)
    dataset_ray = ray.data.from_pandas(dataset_prompts_df)

    return dataset_ray


def prepare_batch_mapper(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(batch):
        ret = tokenizer(list(batch["text"]), padding=True, truncation=True, return_tensors="np")
        return dict(ret)

    batch_mapper = BatchMapper(preprocess_function, batch_format="pandas")

    return batch_mapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-cpu", action="store_true", default=False, help="Enables CPU training")
    parser.add_argument("--no-deepspeed", action="store_true", default=False, help="Disables DeepSpeed strategy")
    parser.add_argument('--model', action='store', type=str, default="tiiuae/falcon-7b", help='Model from HuggingFace to use')
    parser.add_argument('--data', action='store', type=str, default="alpaca_data_cleaned_spanish.json", help='Path of the data to use for finetuning the model')

    args, _ = parser.parse_known_args()

    # Get DeepSpeed config
    if not args.no_deepspeed:
        with open("../config/deepspeed.json","r") as fp:
            deepspeed = json.load(fp)

    # Init Ray cluster
    ray.init(address="auto")
    pprint(ray.cluster_resources())

    # Prepare Ray dataset and batch mapper
    dataset = prepare_dataset(f"../data/{args.data}", args.model)
    batch_mapper = prepare_batch_mapper(args.model)

    # Trainer init config
    trainer_init_config = {"model": args.model,
                           "lora": {"r": LORA_R,
                                    "alpha": LORA_ALPHA,
                                    "dropout": LORA_DROPOUT,
                                    "target_modules": LORA_TARGET_MODULES
                                    },
                            "platform": {"use_gpu": not args.use_cpu,
                                         "deepspeed": deepspeed if not args.no_deepspeed else None}
                            }

    trainer = TransformersTrainer(
        trainer_init_per_worker=trainer_init_per_worker,
        trainer_init_config=trainer_init_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=not args.use_cpu),
        datasets={
            "train": dataset
        },
        preprocessor=batch_mapper,
    )

    # Launch the training on the cluster
    result = trainer.fit()

import os
from transformers import DataCollatorForSeq2Seq, T5Tokenizer
from datasets import arrow_dataset
from torch.utils.data import DataLoader
import datasets
from torch.utils.data import Subset
import torch
from functools import partial
import pandas as pd
import ast
import json
from shifts import (
    label_shift, covariate_shift_typos, label_shift_partial
    )
import random


class Task:
    def __init__(self, task_name, tokenizer, soft_labels, with_shift, seed, perc_rand, shift_order):
        self.task_name = task_name
        self.seed = seed
        self.perc_rand = perc_rand
        self.shift_order = shift_order
        self.with_shift = with_shift
        DATA_DIR = os.getenv("DATA_PATH")
        self.path = os.path.join(DATA_DIR, task_name)
        self.tokenizer = tokenizer
        self.load_config()
        self.load_data()
        self.soft_labels = soft_labels

    def load_config(self):
        with open(os.path.join(self.path, "config.json")) as f:
            config = json.load(f)
        self.is_classification = config["is_classification"]
        if self.is_classification:
            self.classes = config["classes"]
            self.soft_classes = config["soft_classes"]
        self.data_path = config["training_data"]
        if self.with_shift == 'covariate':
            self.typos_data_path = config["typos_data"]

    def load_data(self):
        data = {}
        for split, split_path in self.data_path.items():
            fin = pd.read_csv(
                os.path.join(self.path, split_path),
            )
            fin = apply_data_distr_shift(self, fin, split)
            print(fin["gold_hard"])
            inputs = list(fin["input"].values.astype(str))
            gold_hard = list(fin["gold_hard"].values.astype(str))
            if "llm_soft" in fin.columns:
                llm_soft = list(fin["llm_soft"].values.astype(str))
            if "llm_hard" in fin.columns:
                llm_hard = list(fin["llm_hard"].values.astype(str))

            data[split] = arrow_dataset.Dataset.from_dict(
                {
                    "inputs": inputs,
                    "gold_hard": gold_hard,
                    "llm_hard": llm_hard,
                    "llm_soft": llm_soft,
                }
            )
        self.raw_data = datasets.DatasetDict(data)

    def load_classes(self):
        self.classes_dict = {}
        self.classes_dict_gold = {}
        for idx, class_name in enumerate(self.classes):
            target = self.tokenizer.encode(class_name, add_special_tokens=False)[0]
            self.classes_dict[self.soft_classes[idx]] = target
            self.classes_dict_gold[class_name] = target
        return

    def preprocess(self, accelerator, args, model=None):
        def process_data_to_model_inputs(is_eval: bool, batch):
            out = {}
            # Tokenizer will automatically set [BOS] <text> [EOS]
            out["input_ids"] = self.tokenizer(
                batch["inputs"],
                padding=False,
                max_length=args.max_length,
                truncation=True,
            ).input_ids

            if self.is_classification:
                out["gold_soft"] = make_soft(batch["gold_hard"], target="gold")
                if not self.soft_labels:
                    out["llm_soft"] = make_soft(batch["llm_hard"], target="llm")
                else:
                    out["llm_soft"] = select_classes(batch["llm_soft"])
                    
            if is_eval:
                out["gold_hard"] = batch["gold_hard"]
                out["llm_hard"] = batch["llm_hard"]
            else:
                # limited to max_out_length
                out["gold_hard"] = self.tokenizer(
                    batch["gold_hard"],
                    padding=False,
                    max_length=args.max_out_length,
                    truncation=True,
                ).input_ids
                out["llm_hard"] = self.tokenizer(
                    batch["llm_hard"],
                    padding=False,
                    max_length=args.max_out_length,
                    truncation=True,
                ).input_ids
            return out

        def collate_for_eval(default_collate, batch):
            inputs = [{"input_ids": x["input_ids"]} for x in batch]
            out = default_collate(inputs)
            out["llm_hard"] = [x["llm_hard"] for x in batch]
            out["gold_hard"] = [x["gold_hard"] for x in batch]
            if self.is_classification:
                out["llm_soft"] = [x["llm_soft"] for x in batch]
                out["gold_soft"] = [x["gold_soft"] for x in batch]
            return out

        def select_classes(batch_soft_labels):
            new_batch = []
            for soft_labels in batch_soft_labels:
                soft_labels = ast.literal_eval(soft_labels)
                soft_labels = soft_labels[0]
                new_soft_labels = []
                for key in self.soft_classes:
                    if key in soft_labels:
                        new_soft_labels.append(soft_labels[key])
                    else:
                        new_soft_labels.append(-100)
                new_batch.append(new_soft_labels)
            return new_batch

        def make_soft(batch_hard_labels, target):
            if target == "gold":
                classes_dict = self.classes_dict_gold
            else:
                classes_dict = self.classes_dict
            new_batch = []
            for hard_label in batch_hard_labels:
                new_soft_labels = []
                for label in classes_dict.keys():
                    if label == hard_label:
                        new_soft_labels.append(0)
                    else:
                        new_soft_labels.append(-100)
                new_batch.append(new_soft_labels)
            return new_batch

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=model, padding="longest"
        )
        eval_collator = partial(collate_for_eval, data_collator)

        processed_data = {}

        for split in self.data_path.keys():
            print('----Split: ', split)
            max_samples = getattr(args, f"{split}_samples")
            self.raw_data[split] = random_subset(
                dataset=self.raw_data[split],
                max_samples=max_samples,
                seed=args.seed,
            )

            self.raw_data[split] = arrow_dataset.Dataset.from_list(
                list(self.raw_data[split])
            )
            processed_data[split] = self.raw_data[split].map(
                partial(process_data_to_model_inputs, split in ["test"]),
                batched=True,
                batch_size=args.per_device_eval_batch_size,
                remove_columns=self.raw_data[split].column_names,
            )

        online_dataloader = DataLoader(
            processed_data["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=1,
        )

        test_dataloader = DataLoader(
            processed_data["test"],
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        test_wrong = self.find_wrong_LLM_predictions(processed_data)

        test_wrong_dataloader = DataLoader(
            test_wrong,
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        online_dataloader, test_dataloader, test_wrong_dataloader = accelerator.prepare(
            online_dataloader,
            test_dataloader,
            test_wrong_dataloader,
        )

        self.data = {
            "online_dataloader": online_dataloader,
            "test_dataloader": test_dataloader,
            "test_wrong_dataloader": test_wrong_dataloader,
        }
        return

    def find_wrong_LLM_predictions(self, processed_data):
        '''
        Find the test samples where the LLMs prediction doesn't match the gold label
        '''
        idx_wrong = []
        for idx in range(len(processed_data["test"])):
            tgt = (processed_data["test"]["gold_soft"][idx]).index(
                max(processed_data["test"]["gold_soft"][idx])
            )
            llm_pred = (processed_data["test"]["llm_soft"][idx]).index(
                max(processed_data["test"]["llm_soft"][idx])
            )
            if tgt != llm_pred:
                idx_wrong.append(idx)
        return processed_data["test"].select(idx_wrong)

def random_subset(dataset, max_samples: int, seed: int = 42):
    '''
    Random subset of the dataset maintaining its original order
    '''
    if max_samples >= len(dataset) or max_samples == -1:
        return dataset

    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Randomly sample indices and sort them to maintain the original order
    selected_indices = sorted(random.sample(range(len(dataset)), max_samples))
    
    # Return the subset containing the selected indices
    return Subset(dataset, selected_indices)

def get_task(accelerator, args, model=None):
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length
    )

    # load config, data, and preprocess
    task = Task(args.task_name, tokenizer, args.soft_labels, args.with_shift, args.seed, args.perc_rand, args.shift_order)
    if task.is_classification:
        task.load_classes()
    task.preprocess(accelerator, args, model=None)
    return task


def make_datacollator(args, tokenizer, processed_data, model=None):
    processed_data = arrow_dataset.Dataset.from_dict(processed_data)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    aux = processed_data.train_test_split(test_size=0.1)
    train_dataloader = DataLoader(
        aux["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        aux["test"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, eval_dataloader

def apply_data_distr_shift(self, fin, split):
    # Perform synthetic dataset operations.
    print(self.with_shift)
    if self.with_shift=='label':
        fin = label_shift(fin, self.shift_order)
        print('label')
    if self.with_shift=='label-shift-partial':
        fin = label_shift_partial(fin, self.perc_rand, self.shift_order, self.seed)
        print('label_shift_partial', self.perc_rand)
    if self.with_shift=='covariate':
        print('Split:', split)
        typos_path = os.path.join(self.path, self.typos_data_path[split])
        fin = covariate_shift_typos(fin, self.perc_rand, typos_path, self.shift_order, self.seed)
        print('covariate shift typos')
        print(fin['input'])
    return fin
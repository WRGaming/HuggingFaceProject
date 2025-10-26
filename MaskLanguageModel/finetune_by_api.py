from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator, TrainingArguments, Trainer
from datasets import load_dataset
import collections
import numpy as np
from torch.utils.data import DataLoader
import math

wwm_probability = 0.2


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

    return result


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        mapping = collections.defaultdict(list)
        current_word_id = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id != current_word:
                current_word = word_id
                current_word_id += 1
            mapping[current_word_id].append(idx)

        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        inputs_ids = feature["input_ids"]
        labels = feature["labels"]

        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                inputs_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = whole_word_masking_data_collator(features)

    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

imdb_dataset = load_dataset("imdb")

tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])

chunk_size = 128

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

eval_dataset = lm_datasets.map(
    insert_random_mask,
    batched=True,
    remove_columns=lm_datasets["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
batch_size = 64
train_dataloader = DataLoader(
    lm_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=whole_word_masking_data_collator,
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    collate_fn=default_data_collator
)
logging_steps = len(lm_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}--finetuned-imdb",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    remove_unused_columns=False,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=whole_word_masking_data_collator,
    tokenizer=tokenizer
)

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

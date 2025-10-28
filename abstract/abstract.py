from datasets import load_dataset, concatenate_datasets, DatasetDict
from nltk import sent_tokenize
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
import evaluate
import numpy as np


def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}'")


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_predictions]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = rouge_score.compute(
        predictions=decoded_predictions, refernces=decoded_labels, use_stemmer=True,
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


spanish_dataset = load_dataset("neonwatty/amazon_reviews_multi", "es")
english_dataset = load_dataset("neonwatty/amazon_reviews_multi", "en")

concatenated_dataset = DatasetDict()

for split in english_dataset.keys():
    concatenated_dataset[split] = concatenate_datasets(
        [english_dataset[split], spanish_dataset[split]]
    )
    concatenated_dataset[split] = concatenated_dataset[split].shuffle(seed=42)

concatenated_dataset = concatenated_dataset.filter(lambda x: len(x["review_title"].split()) > 2)

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# max_input_length = 512
# max_target_length = 30
#
# tokenized_datasets = concatenated_dataset.map(preprocess_function, batched=True)
#
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
#
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# tokenized_datasets = tokenized_datasets.remove_columns(
#     concatenated_dataset["train"].column_names
# )
#
# batch_size = 8
# num_train_epochs = 8
#
# logging_steps = len(tokenized_datasets["train"]) // batch_size
# model_name = model_checkpoint.split("/")[-1]
#
# args = Seq2SeqTrainingArguments(
#     output_dir=f"{model_name}-finetuned-amazon-en-es",
#     eval_strategy="epoch",
#     learning_rate=5.6e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=num_train_epochs,
#     predict_with_generate=True,
#     logging_steps=logging_steps,
# )
#
# rouge_score = evaluate.load("rouge")
#
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
#
# print(trainer.evaluate())

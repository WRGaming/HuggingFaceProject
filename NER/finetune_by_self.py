from transformers import AutoModelForTokenClassification, get_scheduler
from NER.finetune_by_api import id2label, label2id, label_names, tokenizer
from finetune_by_api import data_collator, tokenized_datasets, model_checkpoint, metric
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
import torch

output_dir = "bert-finetuned-ner-self"

def postprocess(predictions, labels):  # 评估
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # 清除特殊token(-100)并转换为标签
    true_labels = [[label_names[i] for i in label if i != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, true_labels)
    ]

    return true_labels, true_predictions


# 自定义dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=8,
)

# 定义model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, id2label=id2label, label2id=label2id)

# 定义optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 定义scheduler
num_training_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_update_steps_per_epoch * num_training_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 设置分布式计算
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_training_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # 由于tokenizer会改变dataset形状，需要pad
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    # 打包保存
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

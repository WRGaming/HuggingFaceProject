from transformers import pipeline

from abstract import concatenated_dataset

hub_model_id = "mt5-small-finetuned-amazon-en-es/checkpoint-28500"
summarizer = pipeline("summarization", model=hub_model_id)

def print_summary(idx):
    review = concatenated_dataset["test"][idx]["review_body"]
    title = concatenated_dataset["test"][idx]["review_title"]
    summary = summarizer(concatenated_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")


print_summary(100)
print_summary(0)
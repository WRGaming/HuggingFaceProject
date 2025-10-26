from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "bert-finetuned-ner-self"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))

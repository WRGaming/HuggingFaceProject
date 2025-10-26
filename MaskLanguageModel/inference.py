from transformers import pipeline

# 1. 加载模型
mask_filler = pipeline(
    "fill-mask", model="distilbert-base-uncased--finetuned-imdb/checkpoint-2874"
)

text = "This is a great [MASK]."
# 2. inference
preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")

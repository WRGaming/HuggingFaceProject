from transformers import pipeline

model_checkpoint = "bert-finetuned-squad/checkpoint-33276"

question_answer = pipeline("question-answering", model=model_checkpoint)

context = """
  Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

question = "Which deep learning libraries back   Transformers?"

print(question_answer(question, context))

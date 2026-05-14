import torch
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def tokenize_text(tokenizer, text):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )


def compute_perplexity(model, tokenizer, dataset, format_fn):
    model.eval()
    losses = []

    for example in dataset:
        text = format_fn(example)
        inputs = tokenize_text(tokenizer, text)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(
                **inputs,
                labels=inputs["input_ids"]
            )
            loss = outputs.loss

        losses.append(loss.item())

    ppl = torch.exp(torch.tensor(losses).mean()).item()
    return ppl
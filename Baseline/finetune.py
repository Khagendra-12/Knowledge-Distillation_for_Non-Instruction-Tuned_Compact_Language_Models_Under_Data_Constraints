from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, OUTPUT_DIR, SFT_MODEL_PATH, BATCH_SIZE, EPOCHS, LR, MAX_LENGTH
from dataset import load_dolly
from utils import load_tokenizer

def preprocess_dataset(dataset, tokenizer):
    def tokenize(example):
        if example.get("context"):
            prompt_text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['context']}\n\n### Response:\n"
        else:
            prompt_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"

        response_text = example["response"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) + len(response_ids) > MAX_LENGTH:
            allowed_response_len = MAX_LENGTH - len(prompt_ids)
            if allowed_response_len > 10:
                response_ids = response_ids[:allowed_response_len-1] + [tokenizer.eos_token_id]
            else:
                prompt_ids = prompt_ids[:MAX_LENGTH - 21]
                response_ids = response_ids[:20] + [tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    train_ds, val_ds = load_dolly()
    tokenizer = load_tokenizer()
    print("Tokenizing dataset...")

    train_ds = preprocess_dataset(train_ds, tokenizer)
    print("Loading model...")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Applying LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=1.0,
        warmup_steps=100,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(SFT_MODEL_PATH)
    tokenizer.save_pretrained(SFT_MODEL_PATH)

if __name__ == "__main__":
    main()
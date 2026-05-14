import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Locutusque/TinyMistral-248M" 
LORA_ADAPTER_PATH = "./outputs/dolly-sft" 
MERGED_OUTPUT_PATH = "./outputs/tinymistral-sft-merged"


def main():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32, 
        device_map="cpu" 
    )

    print(f"Loading LoRA adapters from: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    print("Merging weights... (This permanently fuses the LoRA into the base model)")
    merged_model = model.merge_and_unload()

    print(f"Saving fully merged SFT model to: {MERGED_OUTPUT_PATH}")
    merged_model.save_pretrained(MERGED_OUTPUT_PATH)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

    print("Merge complete!")

if __name__ == "__main__":
    main()
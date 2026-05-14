import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from tqdm import tqdm
import numpy as np
import time
from bert_score import score as bert_score_compute
import transformers
def _fallback_build_inputs(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return token_ids_0
    return token_ids_0 + token_ids_1
transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens = _fallback_build_inputs
transformers.PreTrainedTokenizerFast.build_inputs_with_special_tokens = _fallback_build_inputs
from config import SSFT_MODEL_PATH
from dataset import load_dolly
from utils import load_tokenizer

rouge = load("rouge")
bleu = load("bleu")
exact_match = load("exact_match")
meteor = load("meteor")

def format_prompt(example):
    if example.get("context"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['context']}\n\n### Response:\n"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n"

def compute_perplexity(eval_model, tokenizer, dataset):
    eval_model.eval()
    losses = []
    
    print("Evaluating SFT Perplexity...")
    for example in tqdm(dataset):
        full_text = format_prompt(example) + example["response"] + tokenizer.eos_token
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(eval_model.device)
        
        with torch.no_grad():
            safe_input_ids = inputs["input_ids"].clone()
            eval_vocab_size = eval_model.config.vocab_size
            safe_input_ids[safe_input_ids >= eval_vocab_size] = 0 
            
            outputs = eval_model(
                input_ids=safe_input_ids, 
                attention_mask=inputs["attention_mask"], 
                labels=safe_input_ids
            )
            losses.append(outputs.loss.item())
                
    return np.exp(np.mean(losses))

def generate_response_with_latency(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    safe_input_ids = inputs["input_ids"].clone()
    safe_input_ids[safe_input_ids >= model.config.vocab_size] = 0
    inputs["input_ids"] = safe_input_ids

    safe_pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id < model.config.vocab_size else 0

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            num_beams=4, 
            early_stopping=True, 
            pad_token_id=safe_pad_token_id
        )
    latency = time.time() - start_time
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip(), latency, len(generated_tokens) / latency if latency > 0 else 0

def compute_all_metrics(eval_model, tokenizer, dataset):
    preds, refs, latencies, tps_list = [], [], [], []
    print("Generating responses for Comprehensive Evaluation...")
    for example in tqdm(dataset):
        prediction, latency, tps = generate_response_with_latency(eval_model, tokenizer, format_prompt(example))
        preds.append(prediction)
        refs.append(example["response"].strip())
        latencies.append(latency)
        tps_list.append(tps)

    rouge_res = rouge.compute(predictions=preds, references=refs)
    em_res = exact_match.compute(predictions=preds, references=refs)

    print("Calculating BERTScore (Running on CPU to prevent VRAM crashes)...")
    try:
        P, R, F1 = bert_score_compute(
            preds, 
            refs, 
            lang="en", 
            model_type="roberta-large", 
            device="cpu",      
            batch_size=8,      
            verbose=False
        )
        avg_feature_sim = F1.mean().item()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Native BERTScore failed because: {e}\n")
        avg_feature_sim = "Error"
    
    return {
        "rouge": rouge_res,
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs]),
        "meteor": meteor.compute(predictions=preds, references=refs)["meteor"],
        "exact_match": em_res["exact_match"],
        "feature_similarity": avg_feature_sim,
        "avg_latency": np.mean(latencies), 
        "avg_tokens_per_sec": np.mean(tps_list)
    }

def main():
    print("Loading dataset...")
    _, val_ds = load_dolly()
    val_ds_small = val_ds.select(range(100))

    print(f"Loading SSFT model from: {SSFT_MODEL_PATH}")
    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(SSFT_MODEL_PATH, torch_dtype=torch.float32, device_map="auto")
    ppl = compute_perplexity(model, tokenizer, val_ds)
    metrics = compute_all_metrics(model, tokenizer, val_ds_small)

    print("SFT (Soft) MODEL RESULTS")
    print(f"Perplexity:         {ppl:.4f}")
    
    print("\n--- Structural Metrics ---")
    print(f"ROUGE-1:            {metrics['rouge']['rouge1']:.4f}")
    print(f"ROUGE-2:            {metrics['rouge']['rouge2']:.4f}")
    print(f"ROUGE-L:            {metrics['rouge']['rougeL']:.4f}")
    print(f"BLEU:               {metrics['bleu']['bleu']:.4f}")
    print(f"METEOR:             {metrics['meteor']:.4f}")
    print(f"Accuracy (EM):      {metrics['exact_match']:.4f}")
    
    print("\n--- Semantic Metrics ---")
    if isinstance(metrics['feature_similarity'], str):
        print(f"Feature Sim (F1):   {metrics['feature_similarity']}")
    else:
        print(f"Feature Sim (F1):   {metrics['feature_similarity']:.4f} (BERTScore)")
        
    print("\n--- Performance Metrics ---")
    print(f"Avg Latency:        {metrics['avg_latency']:.2f} s/prompt")
    print(f"Speed:              {metrics['avg_tokens_per_sec']:.2f} tokens/s")

if __name__ == "__main__":
    main()
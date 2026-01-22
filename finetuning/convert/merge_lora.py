import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--lora_path', type=str)
    parser.add_argument('--saved_tokenizer_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    # Load base
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.lora_path)

    # Merge LoRA into base
    model = model.merge_and_unload()

    # Save merged HF model
    model.save_pretrained(args.output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.saved_tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    print("LoRA merged and saved to:", args.output_path)

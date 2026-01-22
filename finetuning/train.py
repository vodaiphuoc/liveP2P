import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, default_data_collator
from transformers.trainer_utils import is_main_process
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments
import torch
from torch.utils.data import Dataset
import argparse
import json

from utils.phonemize_text import phonemize_with_dict

def preprocess_sample(sample, tokenizer, max_len=2048):
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100
    
    phones = sample["phones"]
    vq_codes = sample["codes"]
    
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    
    ids = tokenizer.encode(chat)
    
    # Pad/truncate
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    elif len(ids) > max_len:
        ids = ids[:max_len]
    
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, ignore_index)
    
    # Mask labels before speech generation
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

class VieNeuDataset(Dataset):
    def __init__(self, 
                 encoded_data: list[dict[str,str|int]], 
                 tokenizer, 
                 max_len=2048
                ):
        self.samples = encoded_data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["transcript"]
        
        try:
            phones = phonemize_with_dict(text)
        except Exception as e:
            print(f"⚠️ Phonemization error: {e}")
            phones = text
        
        data_item = {"phones": phones, "codes": sample["codes"]}
        return preprocess_sample(data_item, self.tokenizer, self.max_len)


def get_training_args(config):
    return TrainingArguments(
        output_dir=os.path.join(config['output_dir'], config['run_name']),
        do_train=True,
        do_eval=True,
        # max_steps=config['max_steps'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type = config['lr_scheduler_type'],
        warmup_ratio=config['warmup_ratio'],
        bf16=config['bf16'],
        
        # --- CẤU HÌNH QUAN TRỌNG CHO T4 ---
        fp16=True,                           # T4 chạy fp16 rất tốt
        gradient_checkpointing=True,         # Bắt buộc bật để tiết kiệm RAM
        gradient_checkpointing_kwargs={'use_reentrant': False},
        
        # Tối ưu bộ nhớ & Tốc độ
        group_by_length=False,                # CỰC KỲ QUAN TRỌNG: Gom các câu cùng độ dài train chung để đỡ tốn RAM padding
        optim="adamw_torch",                 # Optimizer ổn định
        ddp_find_unused_parameters=False,    # Tắt cái này để tránh lỗi khi chạy nhiều GPU
        # ----------------------------------
        
        logging_steps=config['logging_steps'],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        # dataloader_num_workers=2,
        # dataloader_prefetch_factor= 1,
        # dataloader_pin_memory=True,
        use_liger_kernel = True
    )


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Training Config (Tối ưu cho T4 x2)
training_config = {
    'model': "pnnbao-ump/VieNeu-TTS-0.3B",
    'run_name': "VieNeu-TTS-LoRA",
    'output_dir': "output",
    
    # --- CẤU HÌNH CHO TUAL T4 ---
    'per_device_train_batch_size': 4,   # Giữ là 1 để an toàn vì VRAM T4 (15GB) < P100 (16GB)
    'gradient_accumulation_steps': 4,   # Tăng lên 8 (để bù lại batch size nhỏ)
    # ----------------------------
    
    'learning_rate': 2e-4,
    'lr_scheduler_type': "cosine",
    'warmup_ratio': 0.05,
    'logging_steps': 50,
    'bf16': False,
}

def main(encoded_data_path:str):
    DATA_ENCODED = None
    assert os.path.isfile(encoded_data_path)

    with open(encoded_data_path,"r", encoding='utf-8') as fp:
        DATA_ENCODED = json.load(fp)

    # trick
    for ele in DATA_ENCODED:
        phonemize_with_dict(ele['transcript'])

    # for debug only
    DATA_ENCODED = DATA_ENCODED[:300]

    # Lấy tên model từ config đã khai báo ở cell trước
    model_name = training_config['model']
    print(f"Loading model: {model_name}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model (SỬA QUAN TRỌNG CHO T4)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # T4 tối ưu cho float16, không dùng bfloat16
        dtype=torch.float16,
        # Tự động chia model sang 2 GPU để tận dụng 30GB VRAM gộp
        device_map="auto",
        attn_implementation="sdpa"        
    )

    # 3. Bật tiết kiệm VRAM (BẮT BUỘC để không bị OOM)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # 4. Load Dataset
    full_dataset = VieNeuDataset(DATA_ENCODED, tokenizer)

    # 5. Train/Eval split (5%)
    val_size = max(1, int(0.05 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # 6. Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Khởi tạo Trainer
    # Lưu ý: Hàm get_training_args() phải là hàm MỚI (bản T4) mình đưa ở câu trả lời trước
    args = get_training_args(training_config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 8. Bắt đầu Train
    print('start training')
    trainer.train(resume_from_checkpoint=False)

    # 9. Lưu Model
    if is_main_process():
        save_path = os.path.join(training_config['output_dir'], training_config['run_name'])
        print(f"Saving model to: {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoded_data_file', type=str)
    args = parser.parse_args()

    main(encoded_data_path = args.encoded_data_file)
NLP/NLP03/chatgptbot/scripts/05_inference.py 파일 내용을 전부 지우고 아래 코드로 교체해줘:

import torch
import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import importlib.util
spec = importlib.util.spec_from_file_location('train', os.path.join(os.path.dirname(__file__), '04_train.py'))
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
GPTModel = train_module.GPTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(config.TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

if isinstance(tokenizer, dict):
    word2idx = tokenizer['word2idx']
    idx2word = tokenizer['idx2word']
else:
    word2idx = tokenizer.word2idx
    idx2word = tokenizer.idx2word

vocab_size = len(word2idx)
model_config = {
    'vocab_size': vocab_size,
    'd_model': config.TRANSFORMER_D_MODEL,
    'nhead': config.TRANSFORMER_NHEAD,
    'num_layers': config.TRANSFORMER_NUM_LAYERS,
    'dim_feedforward': config.TRANSFORMER_DIM_FEEDFORWARD,
    'dropout': 0.0,
    'max_seq_length': config.MAX_SEQ_LENGTH
}

model = GPTModel(model_config)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

print(model)

def generate(prompt, max_new_tokens=30):
    tokens = prompt.split()
    input_ids = [word2idx.get(t, word2idx.get('<unk>', 1)) for t in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_tensor.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
            logits = model(input_tensor, None, causal_mask)
            next_token = logits[0, -1, :].argmax().item()
            if idx2word.get(next_token) in ['<end>', '<pad>']:
                break
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]]).to(device)], dim=1)
    generated = [idx2word.get(i, '<unk>') for i in input_tensor[0].tolist()]
    return ' '.join(generated)

test_prompts = ['오늘 기분이', '밥 먹었어', '너무 힘들어']
for p in test_prompts:
    print(f"입력: {p}")
    print(f"출력: {generate(p)}")
    print()
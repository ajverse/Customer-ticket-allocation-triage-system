import torch
import torch.ao.quantization
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
import warnings

class TriageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def extract_bert_embeddings(texts, model_name, batch_size=32, device='cpu'):
    print(f"Extracting embeddings using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
    return np.vstack(all_embeddings)

def quantize_model(model_path, save_path):
    print(f"Quantizing model from {model_path}...")
    
    # FIX: Add weights_only=False to allow loading the full model class
    # Since this is a locally trained model, it is safe.
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # Apply dynamic quantization
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if hasattr(torch.ao.quantization, 'quantize_dynamic'):
             quantized_model = torch.ao.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
        else:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
    
    torch.save(quantized_model, save_path)
    print(f"Quantized model saved to {save_path}")
    
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Original Size: {orig_size:.2f} MB | Quantized Size: {quant_size:.2f} MB")
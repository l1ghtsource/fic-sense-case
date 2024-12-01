import torch
import pandas as pd
import logging
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime


@dataclass
class Config:
    output_dir: str = 'output'
    checkpoint: str = 'cointegrated/rubert-tiny2'
    hdim: int = 312
    num_labels: int = 2
    max_length: int = 512
    optim_type: str = 'adamw_torch'
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 32
    n_epochs: int = 5
    lr: float = 5e-5
    warmup_steps: int = 60

class RuBERTInferenceModel:
    def __init__(self, config=None, model_path=None):
        logging.info("Инициализация модели RuBERT...")
        self.config = config if config is not None else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.checkpoint,
            num_labels=self.config.num_labels
        ).to(self.device)
        
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.config.hdim, self.config.hdim // 2),
            torch.nn.Dropout(0.1),
            torch.nn.GELU(),
            torch.nn.Linear(self.config.hdim // 2, self.config.num_labels),
        ).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)) 
        
        self.model.eval()
        logging.info("Модель инициализирована успешно.")
    
    def preprocess_dataframe(self, df):
        logging.info("Начало предобработки данных...")
        df = df.fillna('нет информации')
        
        def remove_duplicate_blocks(text: str) -> str:
            lines = text.split('\n')
            unique_blocks = set()
            result_lines = []
            for line in lines:
                if line not in unique_blocks:
                    unique_blocks.add(line)
                    result_lines.append(line)
            return '\n'.join(result_lines)
        
        df['work_experience'] = df['work_experience'].astype(str).apply(remove_duplicate_blocks)
        df['work_experience'] = df['work_experience'].str[:500]
        df['position'] = df['position'].astype(str).str[:100]
        df['key_skills'] = df['key_skills'].astype(str).str[:150]
        df['salary'] = df['salary'].astype(str).str[:100]
        
        df['text'] = (
            'age:\n' + df['age'].astype(str) +
            '\n\ncity:\n' + df['city'].astype(str) +
            '\n\nwork_experience:\n' + df['work_experience'] +
            '\n\nposition:\n' + df['position'] +
            '\n\nkey_skills:\n' + df['key_skills'] +
            '\n\nclient_name:\n' + df['client_name'].astype(str) +
            '\n\nsalary:\n' + df['salary']
        )
        logging.info("Предобработка данных завершена.")
        return df['text']
    
    def predict_dataframe(self, df, batch_size=32):
        logging.info("Начало предсказания...")
        start_time = datetime.now()
        
        processed_texts = self.preprocess_dataframe(df)
        all_probs = []
        
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i+batch_size]
            batch_inputs = self.tokenizer(
                batch_texts.tolist(), 
                padding=True, 
                truncation=True, 
                max_length=self.config.max_length, 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())
        
        end_time = datetime.now() 
        logging.info(f"Предсказание завершено. Время выполнения: {end_time - start_time}")
        return all_probs


def get_bert_prediction(df):
    # train = pd.read_json('/kaggle/input/ebychiy-fic/client_dataset.json')
    model = RuBERTInferenceModel()
    probabilities = model.predict_dataframe(df)
    probabilities
    return probabilities


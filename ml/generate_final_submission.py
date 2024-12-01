import torch
import pandas as pd
from dataclasses import dataclass
from ml.gemma_inference import load_tokenizer_and_model, get_gemma_prediction

# требуется 2хT4, можно запустить на Kaggle
@dataclass
class GemmaConfig:
    gemma_dir = 'unsloth/gemma-2-9b-it-bnb-4bit'
    lora_dir = 'path' # путь к скачанной папке с https://huggingface.co/TheStrangerOne/FIC-SENCE-Gemma-LORA !!!
    max_length = 2048
    batch_size = 4
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deott: bool = True # https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/52759
    
    
gemma_cfg = GemmaConfig()
df = pd.read_json('test_data.json') # укажите путь к вашим тестовым данным
tokenizer, model = load_tokenizer_and_model(gemma_cfg)
_, class_1 = get_gemma_prediction(df,  model, tokenizer, gemma_cfg.batch_size, gemma_cfg.device, gemma_cfg.max_length, gemma_cfg.deott)

submission = pd.DataFrame({
    'id': range(len(class_1)),
    'score': class_1
})

submission.to_csv('submission.csv', index=False)
import torch
import pandas as pd

from transformers import Gemma2ForSequenceClassification, Gemma2Model
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

USE_DEOTT_TOKENIZER = True  # https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596


class CustomGemma2ForSequenceClassification(Gemma2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        self.score = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, config.hidden_size // 2),
            torch.nn.Dropout(0.1),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size // 2, config.num_labels),
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
        )


def tokenize(tokenizer, df, max_length):
    age = [f'\n\n<age>: {t}' for t in df['age']]
    city = [f'\n\n<city>: {t}' for t in df['city']]
    work_experience = [f'\n\n<work_experience>: {t}' for t in df['work_experience']][:-700]
    position = [f'<position>: {t}' for t in df['position']][:100]
    key_skills = [f'\n\n<key_skills>: {t}' for t in df['key_skills']][:200]
    client_name = [f'\n\n<client_name>: {t}' for t in df['client_name']]
    salary = [f'\n\n<salary>: {t}' for t in df['salary']][:200]

    texts = [
        a + c + we + p + ks + cn + s
        for a, c, we, p, ks, cn, s in zip(
            age,
            city,
            work_experience,
            position,
            key_skills,
            client_name,
            salary
        )
    ]

    tokenized = tokenizer(texts, max_length=max_length, truncation=True)

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    return input_ids, attention_mask


def tokenize_deott(tokenizer, df, max_length):
    def prepare_text(age, city, work_experience, position, key_skills, client_name, salary):
        rounds = [
            f'<start_of_turn>age\n{age}\n\ncity\n{city}\n\n'
            + f'work_experience\n{work_experience[:-700]}\n\n'
            + f'position\n{position[:100]}\n\n'
            + f'key_skills\n{key_skills[:200]}\n\n'
            + f'client_name\n{client_name}\n\n'
            + f'salary\n{salary[:200]}<end_of_turn>'
        ]

        tmp = '\n'.join(rounds)
        for k in range(len(rounds)):
            tmp = '\n'.join(rounds[k:])
            if len(tokenizer(tmp)['input_ids']) < max_length:
                break

        return tmp

    texts = [
        prepare_text(a, c, we, p, ks, cn, s)
        for a, c, we, p, ks, cn, s in zip(
            df['age'],
            df['city'],
            df['work_experience'],
            df['position'],
            df['key_skills'],
            df['client_name'],
            df['salary'],
        )
    ]

    tokenized = tokenizer(texts, max_length=max_length, truncation=True)

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    return input_ids, attention_mask


@torch.no_grad()
def get_gemma_prediction(df, model, tokenizer, batch_size, device, max_length):
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
    
    df.work_experience = df.work_experience.apply(remove_duplicate_blocks)
    
    if not USE_DEOTT_TOKENIZER:
        data = pd.DataFrame()
        data['input_ids'], data['attention_mask'] = tokenize(tokenizer, df, max_length)
        data['length'] = data['input_ids'].apply(len)
    else:
        data = pd.DataFrame()
        data['input_ids'], data['attention_mask'] = tokenize_deott(tokenizer, df, max_length)
        data['length'] = data['input_ids'].apply(len)

    class_0, class_1 = [], []

    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        tmp = data.iloc[start_idx:end_idx]
        input_ids = tmp['input_ids'].to_list()
        attention_mask = tmp['attention_mask'].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            padding='longest',
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()

        class_0.extend(proba[:, 0].tolist())
        class_1.extend(proba[:, 1].tolist())

    return class_0, class_1

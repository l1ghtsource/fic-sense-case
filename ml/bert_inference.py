import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RuBERTInferenceModel:
    def __init__(self, config):
        self.config = config
        self.device = self.config.device

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

        self.model.eval()

    def preprocess_dataframe(self, df):
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

        return df['text']

    def predict_dataframe(self, df, batch_size=32):
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

        return all_probs

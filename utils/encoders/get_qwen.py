import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class TextEncoderQwen():
    def __init__(self):
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')

    def last_token_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_embedding(self, tokens):
        max_length = 8192
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings


class QwenTokenizer():
    def __init__(self, task='', query=''):
        self.task = task if task else 'You are a clinical encoder for population health. The input text contains structured patient data, but many critical clinical variables are missing. Your task is to generate an embedding that represents BOTH the confirmed facts AND the major information gaps that affect near-term health outcomes.'
        self.query = query if query else 'Encode this patient profile with emphasis on factors that drive near-future health evolution:'
        
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
        self.input_texts = f'Instruct: {self.task}\nQuery:{self.query}\n'

    def get_token(self, texts):
        max_length = 8192

        input_texts = [self.input_texts + text for text in texts]
        #print(input_texts)
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        #print(batch_dict.keys())
        return batch_dict
        
        
'''            
texts = ['年龄：65岁，性别：男性，诊断：非小细胞肺癌，分期：III期，治疗方案：化疗和放疗，预后：高风险。',
         '年龄：12岁，性别：女性，诊断：乳腺癌，分期：I期，治疗方案：手术和化疗，预后：良好。']

embedding_extractor = TextEncoderQwen(task=task, query=query)
embeddings = embedding_extractor.get_embedding(texts)
print(embeddings.shape, embeddings)

sim = torch.mm(embeddings[0:1], embeddings[1:2].T)  # (batch1, batch2)
print(sim)
'''
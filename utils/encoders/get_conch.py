import torch
from conch.open_clip_custom import create_model_from_pretrained, create_model_from_pretrained, tokenize, get_tokenizer

class TextEncoderCONCH():
    def __init__(self):
        self.model, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="Your token")
        self.tokenizer = get_tokenizer()
        #self.model = self.model.cuda()

    def get_embedding(self, tokenized_prompts):
        #tokenized_prompts = tokenize(texts=text, tokenizer=self.tokenizer).cuda()
        #tokenized_prompts.shape
        with torch.inference_mode():
            text_embedings = self.model.encode_text(tokenized_prompts)
        print(text_embedings.shape)
        return text_embedings

    def get_token(self, text=''):
        tokenized_prompts = tokenize(texts=text, tokenizer=self.tokenizer).cuda()
        
class CONCHTokenizer():
    def __init__(self):
        self.tokenizer = get_tokenizer()

    def get_token(self, text=''):
        tokenized_prompts = tokenize(texts=text, tokenizer=self.tokenizer)
        return tokenized_prompts
        
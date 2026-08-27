
from encoders.get_conch import TextEncoder

text = ['45 years old', 'breast cancer']

text_encoder = TextEncoder()
print(text_encoder.get_text_embeddings(text).shape)
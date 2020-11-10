import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

df = pd.read_json('data/Movies_and_TV_5.json', lines=True)
df = df[['reviewText']]
df = df.rename(columns={'reviewText': 'review'})

index = pd.read_json('data/amt_train.json').index
df = df.loc[index]
df.to_json('data/review.json')

df = pd.read_json('data/review.json')
index = pd.read_json('data/amt_train.json').index
df = df.loc[index].fillna('')

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda:0')
sentences = df['review'].tolist()
sentence_embeddings = model.encode(sentences, batch_size=100, convert_to_tensor=True,
                                   device='cuda:0', num_workers=4)
torch.save(sentence_embeddings, 'data/review_preprocessed.pt')

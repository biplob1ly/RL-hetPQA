import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('stsb-roberta-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
N_GPU = torch.cuda.device_count()
print(f'GPU count: {N_GPU}')
model.to(device)
print(f'Model device: {model.device}')
cos = nn.CosineSimilarity(dim=-1)


def get_sim(sent1, sent2):
    sent1_embedding = model.encode(sent1, convert_to_tensor=True).unsqueeze(0) #.cpu().numpy().reshape(1,-1)
    sent2_embedding = model.encode(sent2, convert_to_tensor=True).unsqueeze(0) #.cpu().numpy().reshape(1,-1)
    print(f' embed device: {sent1_embedding.device}')
    # cosine_sim_wo_norm1 = util.pytorch_cos_sim(sent1_embedding, sent2_embedding).item()
    cosine_sim_wo_norm1 = cos(sent1_embedding, sent2_embedding).item()
    print(f"cosine similarity wo norm: {cosine_sim_wo_norm1}\n")


sent1 = "I opened an account. How to delete that?"
sent2 = "I deleted my account. How to re-open that?"
get_sim(sent1, sent2)

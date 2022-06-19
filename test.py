#%%
import pickle
import torch
import matplotlib.pyplot as plt
with open('test_rgbs.pkl',"rb") as f:
    rgb = pickle.load(f).cpu()
    rgb = rgb.view(50,50,3)
    plt.imshow(rgb)

#%%
with open('test_results.pkl',"rb") as f:
    rgb = pickle.load(f)
    rgb = rgb['rgb_fine'].detach().cpu()
    rgb = rgb.view(50,50,3)
    plt.imshow(rgb)

#%%
import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
# %%

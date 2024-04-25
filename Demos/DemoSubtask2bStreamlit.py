from streamlit_image_select import image_select
import os
from PIL import Image
import streamlit as st
import models

if not hasattr(Image, 'Resampling'):  # Pillow<9.0
   Image.Resampling = Image

import torch
import clip
from utils import extractData
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_data
def get_data(indexes=range(10,21)): # Returns 10 images and their text
    df_val,images_val,labels_val = extractData("./data/subtask2b/validation.json",images_path="./data/subtask2b_images/val")
    images = []
    texts = []
    images_values = list(images_val.values())
    for i in indexes:
        im = images_values[i].convert("RGB")
        images.append(im)
        txt = df_val.iloc[i]['text'] or ""
        texts.append(txt)
    return images,texts


@st.cache_resource
def get_classi_model(path="2B_Model.pt"): # Returns classification model
    classi_model = models.BaseBinaryMLP(1024,512,256,dropout=0.5)
    classi_model.load_state_dict(torch.load(path))
    classi_model.to(device)
    return classi_model


@st.cache_resource
def get_clip_featureExtract(): # Returns feature extractor
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    return clip_model, clip_preprocess


clip_model, clip_preprocess = get_clip_featureExtract()
classi_model = get_classi_model()
images,texts = get_data()

captions = [txt[:40] for txt in texts]
img = image_select(
    label="Select a cat",
    images=images,
    captions=captions,
    return_value="index"
)

# Extract features
current_image = clip_preprocess(images[img]).unsqueeze(0).to(device)  
image_features = clip_model.encode_image(current_image)#.cpu().detach()
text_features = clip_model.encode_text(clip.tokenize(texts[img],truncate=True).to(device))
# Concatenate
concatenatedval = torch.cat((image_features,text_features),dim=1).type(torch.FloatTensor).to(device)

# Pass through model
percentage = torch.nn.functional.sigmoid(classi_model(concatenatedval)).item()


st.image(images[img])

if(percentage > 0.5):
    result = "This meme was determined to be propagandistic with a certainty of : "+str(100*round(percentage,2))+"%"
    st.info(
    result,
    icon="✅",)
else:
    result = "This meme was determined not to be propagandistic with a certainty of : "+ str(100*round(1-percentage,2))+"%"
    st.info(
        result,
        icon="⛔️",)



#⛔️ ✅

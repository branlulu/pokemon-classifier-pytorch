import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import torch
from model import get_model
from torchvision.transforms import transforms
seed = 0
torch.manual_seed(seed)


st.title('Pokemon Classifier')
st.write("This is an app to classify the first 150 Pokemons.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

  # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to mean 0, std 1
    ])
    
    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    with open('pokemon_to_index.json') as data_file:    
        class_indices = json.load(data_file)
    index_to_class = list(class_indices.keys())

    model = get_model()
    model.load_state_dict(torch.load('pytorch_model_4.model', map_location=torch.device('cpu')), strict=False)
    model.eval()

    probabilities = model(image_tensor).detach().numpy()[0]
    index = np.argsort(-probabilities)
    predictions = [index_to_class[i] for i in index]

    st.write("RESULTS...")
    for i in range(3):
        st.write(predictions[i], ", score:", round(probabilities[index[i]], 3))

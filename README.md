# Pokemon Classifier


## About the project

The app was live at http://3.235.176.30:8501/, using a AWS EC2 Instance and TMUX. However, I discovered that Streamlit Sharing hosts Streamlit apps for free, so I switched over. (Sorry, EC2 Instances cost money.) The app can now be accessed live at: https://share.streamlit.io/branlulu/pokemon-classifier-pytorch/main/app.py.

<img width="378" alt="Screen Shot 2021-07-12 at 11 22 13 PM" src="https://user-images.githubusercontent.com/16676830/125385173-01215e00-e368-11eb-9a7d-9a9d1e84ac9e.png">

## Installation

Clone this repo to your desktop and run `pip install -r requirements.txt` to install the dependencies. I recommend setting up a fresh conda environment for PyTorch, if you haven't already. To set up a Python 3.8 environment using Conda, run `conda create --name pytorch python=3.8`. 

## Usage

After you clone this repo to your desktop, go to its root directory. 

Run `streamlit run app.py` to set up the web app locally. By default, Streamlit will run on `localhost:8501`.

## Features

The app uses a convolutional network trained on a database of pokemon images. The neural network is able to classify the first 150 Pokémon from the Video Game and Television series with 60% test accuracy (from a 70-30 training-test split).

The training set is modified from the following Kaggle dataset: https://www.kaggle.com/lantian773030/pokemonclassification.

It is worth noting that due to insufficient data, the model does not support a few Pokémons (e.g. Nidoran). Future work includes gathering and training the model on a more comprehensive dataset. The model is for recreational use only and is not intended to identify creatures other than those specified. 

## License
This project is licensed under the terms of the MIT license.

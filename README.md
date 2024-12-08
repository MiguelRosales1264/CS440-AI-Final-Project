# Final Project for CS440 Intro To AI - Huberman on Exercise

## Team Members
- **Miguel Rosales**
- **Gabriel Schonacher**
- **Thomnas McRoskey**

## Link to the dataset
### https://huggingface.co/datasets/dexaai/huberman_on_exercise


# Introduction - Purpose of the Project

This project is a final project for the CS440 Intro to AI course. The project uses the Huberman on Exercise dataset by **dexa.ai** and includes transcripts from 200+ podcasts
from the Huberman Lab Podcast. The dataset is used to train a model to answer questions about the podcast. The model is a BigramLanguageModel that uses a bigram language model to predict the next word, character, or sentence in the sequence. The model is trained on about 90% of the transcripts from the Huberman Lab Podcast. The goal of the project is to create a model that can answer questions about the podcast based on the text data in the transcripts. The replies are similar to the replies that something like ChatGPT would generate.


### Commands for us to remember
```bash
# what command lines to I use to create virtual environment with pip?
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.txt to save the requirements

# to update the requirements file
pip install --upgrade pip
pip freeze > requirements.txt to save the requirements

#to install the requirements
pip install -r requirements.txt

# to deactivate the virtual environment
deactivate

# to remove the virtual environment
rm -rf venv



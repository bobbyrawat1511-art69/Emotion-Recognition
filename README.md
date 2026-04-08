# Emotion Recognition Project

## Overview
This repository contains an emotion recognition project that combines text and speech analysis to detect emotions. It includes:
- A Jupyter notebook for building and testing a speech-to-text emotion detection pipeline (`Text-Speech_Emotion.ipynb`)
- A Discord bot implementation for emotion classification of text and voice messages (`Code/multilingualfinal_discord.py`)
- A cleaned dataset for training the emotion model (`Go_Emotion_cleaned_dataset/cleaned_goemotions.csv`)
- A model directory for saving trained model artifacts (`model/`)
- A report and presentation materials in the `Report/` folder

## Repository Structure
- `Text-Speech_Emotion.ipynb` - Notebook with setup, data preprocessing, model training, and evaluation steps
- `Code/multilingualfinal_discord.py` - Discord bot source code using scikit-learn, speech recognition, and translation
- `Go_Emotion_cleaned_dataset/` - Dataset folder containing cleaned GoEmotions data
- `model/` - Output folder for saved model files
- `Report/` - Project report and supporting documents
- `Speech-Text_Emotion.pptx` - Presentation file for the project

## Setup
1. Clone the repository.
2. Install Python dependencies.

```bash
pip install discord.py scikit-learn pandas numpy matplotlib SpeechRecognition pydub joblib deep-translator seaborn imbalanced-learn
```

3. Install additional system requirements:
- FFmpeg (required by `pydub`)
- PyAudio (required by `SpeechRecognition`)

On Windows, PyAudio can be installed with:

```bash
pip install pipwin
pipwin install pyaudio
```

## Configuration
Update the configuration variables in `Code/multilingualfinal_discord.py` or the notebook before running:
- `DISCORD_TOKEN` - your Discord bot token
- `DATASET_PATH` - path to `Go_Emotion_cleaned_dataset/cleaned_goemotions.csv`
- `MODEL_PATH` - path where the trained model should be saved, e.g. `model/emotion_model.pkl`

## Usage
### Notebook
- Open `Text-Speech_Emotion.ipynb` in Jupyter Notebook or JupyterLab
- Run the cells to install dependencies, preprocess the dataset, train the emotion classifier, and evaluate results

### Discord Bot
- Configure the bot token, dataset path, and model path in `Code/multilingualfinal_discord.py`
- Run the script to start the Discord bot

## Notes
- The project uses logistic regression with TF-IDF features and SMOTE for class balancing
- Emotion classes include: `happy`, `sad`, `anger`, `disgust`, `surprise`, `fear`, and `neutral`
- Voice messages are transcribed with Google Speech Recognition and translated to English using `deep-translator`

## Contact
For questions or improvements, update the repository or contact the project maintainer.

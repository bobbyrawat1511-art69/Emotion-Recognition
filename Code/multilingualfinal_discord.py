#Import all the necessary libraries first
import discord
from discord.ext import commands
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from pydub import AudioSegment
import os
import logging
import asyncio
import joblib
from deep_translator import GoogleTranslator
import seaborn as sns

# ----------------------
# CONFIGURATION
# ----------------------
DISCORD_TOKEN = "Discord Bot tokens can be taken from Developer portal: https://discord.com/developers/docs/intro"
DATASET_PATH = r"Enter your Folder path here\cleaned_goemotions.csv"
MODEL_PATH = r"Folder path\model(sub_folder)\emotion_model.pkl"

# Logging
logging.basicConfig(level=logging.INFO)

# ----------------------
# EMOTION CLASSIFIER
# ----------------------
class EmotionClassifier:
    EMOTION_EMOJI_MAP = {
        "happy": "😊",
        "sad": "😢",
        "anger": "😡",
        "fear": "😨",
        "disgust": "🤢",
        "surprise": "😲",
        "neutral": "😐"
    }

    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load_and_train(self):
        df = pd.read_csv(DATASET_PATH)
        emotion_columns = ['happy', 'sad', 'anger', 'disgust', 'surprise', 'fear', 'neutral']
        df['emotion'] = df[emotion_columns].idxmax(axis=1)
        df.dropna(subset=['text', 'emotion'], inplace=True)
        X = df['text']
        y = df['emotion']
        vectorizer = TfidfVectorizer(max_features=10000)
        X_tfidf = vectorizer.fit_transform(X)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        self.model = Pipeline([
            ('clf', LogisticRegression(max_iter=1000))
        ])
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_estimator(self.model, X_test, y_test)
        plt.title('Confusion Matrix')
        plt.show()
        joblib.dump({'model': self.model, 'vectorizer': vectorizer}, MODEL_PATH)
        logging.info("Model trained and saved successfully.")

    def predict(self, text: str):
        if not self.model or not self.vectorizer:
            # Load model and vectorizer from file if not already loaded
            saved_data = joblib.load(MODEL_PATH)
            self.model = saved_data['model']
            self.vectorizer = saved_data['vectorizer']
    
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        confidence = max(self.model.predict_proba(text_vectorized)[0]) * 100
        emoji = self.EMOTION_EMOJI_MAP.get(prediction, "❓")
        return prediction, emoji, confidence


# ----------------------
# VOICE PROCESSOR
# ----------------------
class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    async def process_voice_message(self, file_path: str):
        try:
            audio = AudioSegment.from_file(file_path)
            wav_path = "temp.wav"
            audio.export(wav_path, format="wav")
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            original_text = self.recognizer.recognize_google(audio_data)
            os.remove(wav_path)
            translated_text = GoogleTranslator(source='auto', target='en').translate(original_text)
            return translated_text
        except Exception as e:
            logging.error(f"Error processing voice: {e}")
            return ""

# ----------------------
# DISCORD BOT
# ----------------------
class EmotionBot(commands.Bot):
    def __init__(self, classifier: EmotionClassifier):
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.classifier = classifier
        self.voice_processor = VoiceProcessor()

    async def on_ready(self):
        logging.info(f'Bot connected as {self.user}')

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return
        if message.attachments:
            await self.handle_voice_message(message)
        else:
            text = GoogleTranslator(source='auto', target='en').translate(message.content)
            emotion, emoji, confidence = self.classifier.predict(text)
            if confidence > 90:
                await message.add_reaction(emoji)

    async def handle_voice_message(self, message: discord.Message):
        file_path = 'voice_message.ogg'
        await message.attachments[0].save(file_path)
        text = await self.voice_processor.process_voice_message(file_path)
        os.remove(file_path)
        if text:
            emotion, emoji, confidence = self.classifier.predict(text)
            if confidence > 75:
                await message.add_reaction(emoji)
            logging.info(f"Voice Text: {text} | Emotion: {emotion} | Confidence: {confidence:.2f}%")

# ----------------------
# MAIN FUNCTION
# ----------------------
async def main():
    classifier = EmotionClassifier()
    classifier.load_and_train()
    bot = EmotionBot(classifier)
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())

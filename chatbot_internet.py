import streamlit as st
import requests
from dotenv import load_dotenv
import json
from textblob import TextBlob
import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile

# Load environment variables
load_dotenv()

# Induced AI API configuration
INDUCED_AI_API_KEY = os.getenv('INDUCED_AI_API_KEY')
INDUCED_AI_ENDPOINT = "https://api.induced.ai/api/v1"  # Base URL provided in the documentation

# Title of the application
st.title("My Own AI-Powered Assistant!🤖")

# Initialize session state for model and messages
if "model" not in st.session_state:
    st.session_state.model = "Induced AI"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
st.sidebar.title("Settings")

# Custom parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.number_input("Max Tokens", 1, 4000, 1000)

# Save and load chat history
st.sidebar.title("Chat History")
if st.sidebar.button("Save Chat History"):
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.messages, f)
    st.sidebar.success("Chat history saved!")

uploaded_file = st.sidebar.file_uploader("Upload Chat History", type=["json"])
if uploaded_file is not None:
    st.session_state.messages = json.load(uploaded_file)
    st.sidebar.success("Chat history loaded!")

# Theme customization
st.sidebar.title("Theme")
theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #333;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Memory management
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.sidebar.success("Chat cleared!")

# Function for sentiment analysis
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return f"Sentiment Polarity: {sentiment.polarity}"

# Function to generate voice output
def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        os.system(f"mpg321 {fp.name}")

# Function to capture voice input using Google Speech Recognition
def capture_voice_input_google():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak now!")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.write("Could not understand audio")
        return None
    except sr.RequestError:
        st.write("Could not request results; check your network connection")
        return None

# Function to make an API call to Induced AI
def generate_response_from_induced_ai(prompt):
    headers = {
        "x-api-key": INDUCED_AI_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(f"{INDUCED_AI_ENDPOINT}/run", headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("text", "Error: No 'text' key in the response")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return "Error: Unable to fetch response from Induced AI"

# Display the microphone icon and make it clickable
def microphone_button():
    if st.button('🎤 Click to Speak'):
        return capture_voice_input_google()
    return None

# Capture user input via text or voice
user_prompt = st.chat_input("Your prompt") or microphone_button()

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Sentiment analysis of user input
    sentiment = analyze_sentiment(user_prompt)
    st.write(sentiment)

    # Generate response from Induced AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        assistant_response = generate_response_from_induced_ai(user_prompt)
        message_placeholder.markdown(assistant_response)

    # Sentiment analysis of model response
    sentiment = analyze_sentiment(assistant_response)
    st.write(sentiment)

    # Convert model response to speech
    text_to_speech(assistant_response)

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
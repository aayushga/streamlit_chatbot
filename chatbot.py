import streamlit as st
import openai
from dotenv import load_dotenv
import json
from textblob import TextBlob
import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile
import threading

# Load environment variables
load_dotenv()

# Initialize global variables for recording control
recording = False
stop_recording_event = threading.Event()

# Initialize OpenAI client
client = openai
client.api_key = os.getenv('OPENAI_API_KEY')

# Title of the application
st.title("My Own ChatGPT-Powered Assistant!!🤖")

# Define available models and their brief descriptions
models = {
    "gpt-4": "The full version of GPT-4, offering the most powerful and complex outputs.",
    "gpt-3.5-turbo": "Optimized version of GPT-3.5 with lower latency and better performance.",
    "gpt-4o-mini": "GPT-4 mini optimized for lower latency and faster response."
}

# Initialize session state for model and messages
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recording" not in st.session_state:
    st.session_state.recording = False

# Sidebar for model selection and customization
st.sidebar.title("Settings")

# Model selection
st.session_state.model = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: f"{x} - {models[x]}"
)

# Custom parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.number_input("Max Tokens", 1, 4000, 1000)

# Voice Recognition Method Selection
st.sidebar.title("Voice Recognition")
voice_recognition_method = st.sidebar.selectbox(
    "Choose a voice recognition service:",
    ["Google Speech Recognition", "OpenAI Whisper"]
)

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
    sntimt = TextBlob(text).sentiment
    return f"Sentiment Polarity: {sntimt.polarity}"


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


# Function to capture and process voice input using Whisper (OpenAI API)
def capture_voice_input_whisper():
    st.write("Listening... Please speak now!")
    with st.spinner('Recording your voice...'):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio_data = r.listen(source, timeout=10, phrase_time_limit=10)
                if stop_recording_event.is_set():
                    st.write("Recording stopped by user.")
                    return None
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    temp_audio_file.write(audio_data.get_wav_data())
                    temp_audio_file_path = temp_audio_file.name

                with open(temp_audio_file_path, "rb") as audio:
                    transcript = client.audio.transcriptions.create(
                        file=audio,
                        model="whisper-1",
                        response_format="json"
                    )
                    st.write(transcript)  # Debugging: check the content of the transcript
                    return transcript.text  # Accessing the 'text' attribute directly
            except sr.WaitTimeoutError:
                st.write("Listening timed out while waiting for phrase to start.")
            except sr.UnknownValueError:
                st.write("Could not understand the audio.")
            except sr.RequestError as e:

                st.write(f"Could not request results; {e}")


# Start recording
def start_recording():
    st.session_state.recording = True
    stop_recording_event.clear()
    if voice_recognition_method == "OpenAI Whisper":
        return capture_voice_input_whisper()
    else:
        return capture_voice_input_google()


# Stop recording
def stop_recording():
    stop_recording_event.set()
    st.session_state.recording = False
    st.write("Recording stopped.")


# Display Start or Stop button based on the state
if not st.session_state.recording:
    if st.button("🎤 Start Recording"):
        user_prompt = start_recording()
else:
    if st.button("🛑 Stop Recording"):
        stop_recording()

# Capture user input via text if no voice input
if 'user_prompt' not in locals() or user_prompt is None:
    user_prompt = st.chat_input("Your prompt")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Sentiment analysis of user input
    sentiment = analyze_sentiment(user_prompt)
    st.write(sentiment)

    # Generate response from the selected model
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        assistant_response = response.choices[0].message.content
        message_placeholder.markdown(assistant_response)

    # Sentiment analysis of model response
    sentiment = analyze_sentiment(assistant_response)
    st.write(sentiment)

    # Convert model response to speech
    text_to_speech(assistant_response)

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
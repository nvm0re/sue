import os
import signal
from llama_cpp import Llama  # For the LLaMA model
import speech_recognition as sr  # For speech recognition
from gtts import gTTS  # For text-to-speech
import random  # To add variety in prompts
from gpt4all import GPT4All
import time  # For delay after file saving

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # Downloads / loads a 4.66GB LLM
with model.chat_session():
    print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))

# Path to your LLaMA model file
MODEL_PATH = "/Users/andradabaleanu/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # Update this with the correct path to your model

# Load the LLaMA model
print("Loading the LLaMA model...")
llm = Llama(model_path=MODEL_PATH)
print("Model loaded successfully!")

# Positive prompt templates
PROMPTS = [
    "Please generate a positive affirmation for me.",
    "Can you provide a kind and encouraging statement?",
    "I need a motivational and positive phrase.",
    "Say something uplifting and inspiring.",
]

# Predefined affirmations (fallback if the AI response isn't good enough)
PREDEFINED_AFFIRMATIONS = [
    "You are capable of achieving great things.",
    "Every day is a new opportunity to grow and succeed.",
    "You are worthy of love and respect.",
    "Believe in yourself; you have all the power within you.",
    "You are strong, resilient, and capable of overcoming challenges.",
]

def timeout_handler(signum, frame):
    """Timeout handler to prevent freezing during LLaMA model call."""
    raise TimeoutError("The LLaMA model took too long to respond!")

def listen_to_audio():
    """
    Capture audio from the microphone and convert it to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your request...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.WaitTimeoutError:
            print("No input detected. Please try again.")
            return None

def generate_affirmation():
    """
    Generate a positive affirmation using the LLaMA model.
    If the response is not positive, fallback to predefined affirmations.
    """
    prompt = random.choice(PROMPTS)  # Randomly choose a prompt for variety
    print(f"Generating affirmation with prompt: {prompt}")
    
    # Set the timeout for model response
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # Set the timeout to 10 seconds

    try:
        # Generate response using LLaMA model
        response = llm(prompt, max_tokens=50)
        print("Model response received.")

        # Check the structure of the response (for debugging)
        print(f"Full Model Response: {response}")

        # Extract text from response
        affirmation = response["choices"][0]["text"].strip()
        print(f"Raw AI Response: {affirmation}")

        # Check if the response is positive, else fallback
        if any(neg_word in affirmation.lower() for neg_word in ["not", "sad", "bad", "negative", "cannot"]):
            print("Detected non-positive response. Falling back to predefined affirmation.")
            affirmation = random.choice(PREDEFINED_AFFIRMATIONS)
    except TimeoutError:
        print("The request to the model timed out.")
        affirmation = random.choice(PREDEFINED_AFFIRMATIONS)
    finally:
        signal.alarm(0)  # Disable the timeout
    
    return affirmation

def speak_text(text):
    """
    Convert text to speech and play it.
    """
    print(f"Speaking: {text}")
    
    mp3_filename = "response.mp3"
    try:
        # Ensure the file is overwritten every time
        if os.path.exists(mp3_filename):
            os.remove(mp3_filename)  # Remove the old file to overwrite

        # Save audio to MP3 file
        tts = gTTS(text=text, lang='en')
        tts.save(mp3_filename)

        # Wait a little bit to ensure the file is fully written before playing
        time.sleep(1)

        # Check if the file exists and can be accessed
        if os.path.exists(mp3_filename):
            print(f"MP3 file saved: {mp3_filename}")
        else:
            print(f"Failed to save MP3 file: {mp3_filename}")

        # Attempt to play the MP3 file with appropriate system command
        if os.name == 'nt':  # Windows
            print("Playing audio (Windows)...")
            os.system(f"start {mp3_filename}")
        elif os.name == 'posix':  # macOS or Linux
            print("Playing audio (macOS/Linux)...")
            os.system(f"open {mp3_filename}")  # For macOS
            # If you're on Linux, you might need to use:
            # os.system(f"mpg123 {mp3_filename}")
        else:
            print("Unsupported OS for audio playback.")
    except Exception as e:
        print(f"Error while playing audio: {e}")

# Main program loop
if __name__ == "__main__":
    print("Hello! My name is Sue. (Say 'exit' to quit)")
    
    while True:
        user_input = listen_to_audio()
        
        if user_input:
            if "exit" in user_input:
                print("Goodbye! Have a positive day!")
                break

            affirmation = generate_affirmation()
            print(f"Generated Affirmation: {affirmation}")
            speak_text(affirmation)

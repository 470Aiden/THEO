# THEO - AI Voice Assistant for Elderly Tech Support (Technical Help, Explanations, and Optimization)

## Overview

Theo is an intelligent voice-controlled personal assistant designed specifically to help older adults with everyday technology challenges. The assistant combines speech recognition, natural language processing, computer vision, internet search, media control, and task management into a single easy-to-use system.

Unlike traditional voice assistants, Theo  focuses on providing simple, clear, and non-technical support for users who may have limited technological experience.

## Features

### Wake Word Detection

* Hands-free activation using the wake phrase **"Hey Theo "**
* Always listening in low-power mode
* No need to press buttons or use a keyboard

* For training your own custom wakeword models, I recommend using the free colab notebook by the creator: https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb#scrollTo=qgaKWIY6WlJ1

### Speech Recognition

* Online transcription using Groq's Whisper Large V3 Turbo model
* Automatic offline fallback using Faster-Whisper
* Seamless transition between online and offline modes

### Text-to-Speech Responses

* High-quality online voice synthesis through ElevenLabs
* Offline voice generation using Kokoro ONNX
* Natural and responsive spoken feedback

### Online and Offline Operation

Theo  automatically detects internet availability and adapts its functionality:

**Online Mode**

* Groq LLM
* Groq Whisper transcription
* ElevenLabs speech synthesis
* Weather information
* Web search
* Error screen analysis

**Offline Mode**

* Ollama local language model
* Faster-Whisper transcription
* Kokoro speech synthesis

### ☁️ Weather Information

* Retrieves real-time weather conditions
* Reports temperature and weather descriptions
* Supports any city worldwide

### 🎵 Spotify Voice Control

Control Spotify entirely through voice commands:

* Play songs
* Pause and resume playback
* Skip tracks
* Return to previous tracks

### ⏰ Timers and Reminders

* Create countdown timers
* Set recurring reminders
* Cancel existing reminders
* Audio notifications when reminders expire

### 🔍 Internet Search

* Voice-powered web searching
* Retrieves and summarizes search results
* Provides quick answers to user questions

### 🖥️ Computer Error Analysis

A unique accessibility feature designed for non-technical users.

Theo  can:

1. Detect when a user is experiencing a computer problem
2. Ask them to hold the screen up to a camera
3. Capture an image automatically
4. Analyze the error using a multimodal AI model
5. Explain the issue in simple language

### 👴 Elderly-Friendly Design

* Short responses (maximum two sentences)
* Simple instructions
* Minimal technical terminology
* One-step-at-a-time guidance
* Conversational interaction

---

## System Architecture

### Input Layer

* Wake word detection (OpenWakeWord)
* Voice recording (SoundDevice)
* Camera capture (OpenCV)

### Processing Layer

* Speech-to-text conversion
* Language model reasoning
* Tool selection and execution
* Computer vision analysis

### Output Layer

* Text-to-speech generation
* Audio playback
* Notifications and alarms

---

## Technologies Used

### Artificial Intelligence

* Groq API
* Llama 3.3 70B
* Llama 4 Scout Vision Model
* Faster-Whisper
* Ollama
* OpenAI Compatible API

### Speech Processing

* OpenWakeWord
* ElevenLabs
* Kokoro ONNX
* SoundDevice
* PyAudio

### Computer Vision

* OpenCV
* Base64 Image Encoding

### External Services

* OpenWeatherMap API
* Spotify Web API
* DuckDuckGo Search

### Python Libraries

* requests
* httpx
* numpy
* scipy
* soundfile
* threading
* dotenv
* spotipy

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Theo -assistant.git
cd Theo -assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Environment File

Create a file named:

```text
key.env
```

Add the following API keys:

```env
GROQ_API_KEY=your_groq_api_key

ELEVEN_LABS_KEY=your_elevenlabs_key

WEATHER_API_KEY=your_openweathermap_key

SPOTIFY_CLIENT_ID=your_spotify_client_id

SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

---

## Running Theo 

```bash
python main.py
```

Theo  will begin listening for the wake word:

```text
"Hey Theo "
```

After activation, simply speak your request.

---

## Example Commands

### General Questions

```text
Hey Theo , what time is it?
```

```text
Hey Theo , what's the weather in Kingston?
```

### Music Control

```text
Hey Theo , play Shape of You by Ed Sheeran.
```

```text
Hey Theo , pause the music.
```

```text
Hey Theo , skip this song.
```

### Timers

```text
Hey Theo , set a timer for 10 minutes.
```

### Reminders

```text
Hey Theo , remind me to take my medication in 3 hours.
```

```text
Hey Theo , cancel my medication reminder.
```

### Web Search

```text
Hey Theo , search for the latest news about Jamaica.
```

### Technical Support

```text
Hey Theo , I'm getting an error on my computer.
```

Theo  will request that the user show the screen to the camera and then provide a simplified explanation of the problem.





Project Goal

The primary goal of Theo  is to make technology more accessible to elderly users by providing a voice-first interface that reduces reliance on keyboards, touchscreens, and technical knowledge. By combining AI-powered assistance with simple communication, Theo  helps bridge the gap between modern technology and users who may find it difficult to navigate.




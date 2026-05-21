import os
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq 
from dotenv import load_dotenv
from datetime import datetime
import requests
import json
import httpx
from openai import OpenAI
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from elevenlabs.client import ElevenLabs
import elevenlabs.play
import soundfile as sf
import numpy as np
from openwakeword.model import Model
import pyaudio
import spotipy
from spotipy.oauth2 import SpotifyOAuth



load_dotenv("key.env")#uses python-dotenv library to read the .env file for the api key
api_key = os.environ.get("GROQ_API_KEY")  #assigns the api key to a variable
client = Groq(api_key=api_key)
el_client = ElevenLabs(api_key=os.environ.get("ELEVEN_LABS_KEY"))

kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin") #sets up kokoro 
#tries accessing the groq api link. Will not work without internet access.
try:
    httpx.get("https://api.groq.com", timeout=2)
    ONLINE = True
except:
    ONLINE = False

wakeword_model=Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
def listen_for_wake():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=16000,channels=1,
        format=pyaudio.paInt16,
        input=True,frames_per_buffer=1280
    )
    print("Listening for wake word")
    while True:
        audio_chunk = np.frombuffer(stream.read(1280), dtype=np.int16)
        prediction = wakeword_model.predict(audio_chunk)
        if prediction["hey_jarvis"] > 0.5:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            return    # Wake word heard, exit and start recording
def speak(reply):
    if ONLINE == True:
        audio = el_client.text_to_speech.convert(
            text = reply,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128"
        )
        elevenlabs.play.play(audio)
    else:
        samples, sample_rate = kokoro.create(
            reply,
            voice="af_heart",
            speed=1.0,
            lang="en-us"
        )
        sd.play(samples,sample_rate)
        sd.wait()

local_whisper = WhisperModel("base", device="cpu", compute_type="int8")
#Checks internet access to determine whether to use cloud whisper model or local whisper model.
def transcribe(filename):
    if ONLINE == True:
        # Online — use Groq Whisper
        print("[Online - using Groq Whisper]")
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                language="en"
            )
        return transcription.text
    else:
        # Offline — use local Whisper
        print("[Offline - using local Whisper]")
        segments, _ = local_whisper.transcribe(filename, language="en")
        return " ".join([s.text for s in segments])
#Checks internet access to determine whether to use cloud groq model or local ollama model.
def get_client():
    if ONLINE == True:
        return client, "llama-3.3-70b-versatile"
    else:
        print("Offline - Using Ollama")
        local_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        return local_client,"llama3.2:3b"




def get_time():
    return datetime.now().strftime("%I:%M %p") #formats time it terms of hour,minute, AM/PM

def get_weather(city):
    try:
        load_dotenv("key.env")
        weather_key = os.environ.get("WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        temp= data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The current temperature {temp}°C, {description} in {city}"
    except:
        return "Cannot reach weather data, there is no internet available"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
    client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-modify-playback-state user-read-playback-state user-read-recently-played"
))
def play_spotify(query):
    try:
        results = sp.search(q=query, type="track",limit=1)
        track_uri= results["tracks"]["items"][0]["uri"]
        sp.start_playback(uris=[track_uri])
        track_name= results["tracks"]["items"][0]["name"]
        artist = results["tracks"]["items"][0]["artists"][0]["name"]
        return f"Playing {track_name} by {artist}"
    except Exception as e:
        print(f"Spotify error is {e}")
        return "Couldn't play that. Ensure spotify is open on your device."

def pause_spotify():
    try:
        playback= sp.current_playback()
        if playback and playback["is_playing"]:
            sp.pause_playback()
            return "Track paused"
        else:
            sp.start_playback()
            return "Track resumed"
    except:
        return "Nothing is playing"

def next_track():
    try:
        sp.next_track()
        return "Track skipped"
    except:
        return "End of playlist"
def previous_track():
    try:
        sp.previous_track()
        return "Now playing the previous track"
    except:
        return "No previous tracks"

#Describes the tool for the ai, saying "Here is a tool you can use"
tools=[
    {
        "type":"function",

        "function": {
            "name":"get_time",
            "description":"Get the current time",
            "parameters":{
                "type":"object",
                "properties":{}
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"get_weather",
            "description":"Gets current weather data in a specified city, including temperature and a description of conditions",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"The name of the city the user is in. E.g- Kingston, London, etc"
                    }
                },
                "required":["city"] #tells the AI it Has to have this info
            }
        }
    },
        {
        "type":"function",
        "function": {
            "name":"play_spotify",
            "description":"Plays a spotify track based on the song and artist name given by the user. Use this whenever the user asks to play a song, and always listen for song and artist names.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The name of the song the user wants to play and the artist it is by."
                    }
                },
                "required":["query"] #tells the AI it Has to have this info
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"pause_spotify",
            "description":"Pauses/unpauses the spotify track which is currently playing.",
            "parameters":{
                "type":"object",
                "properties":{}
            }
        }
    },
     {
        "type":"function",
        "function": {
            "name":"next_track",
            "description":"Plays the next track in the spotify track queue",
            "parameters":{
                "type":"object",
                "properties":{}
            }
        }
    }
    
]
active_client, model = get_client()
messages=[{"role": "system", #tells the system its role
        "content": """You are a friendly home assistant named Jarvis who specializes in 
tech support for older users with little technological experience.
You have tools available to control Spotify, check weather, and get the time.
IMPORTANT: When you need to use a tool, use it directly — never write out function calls as text.
Always keep responses short and simple — 2 sentences maximum.
Avoid technical jargon. If giving instructions, use one step at a time.
Good response example: 'Your printer may be out of paper. Try opening the paper tray and refilling it.'
Bad response example: 'There are several potential causes for this issue. First, let's diagnose whether it's a connectivity problem, a driver issue, or a hardware malfunction...'"""
    }]

listen_for_wake()
speak("Yes?")
while True:
    filename = os.path.dirname(__file__) +"/userspeech.wav"
    # Step 1 — Record first, THEN transcribe
    print("Recording for 5 seconds... speak now")
    audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="int16")
    sd.wait()  # Waits until recording is fully done before moving on
    write(filename, 16000, audio)
    print("Done recording\n")
    usertext = transcribe(filename)
    print(f"You said: {usertext}")
    if usertext.strip() in [".", ""]:  # silence
        print("Nothing heard, listening again...\n")
        continue
    if "quit" in usertext.lower():  # exit
        break
    #Adds the new message to chat history
    messages.append({"role":"user", "content":usertext})

    response = active_client.chat.completions.create(
        tools=tools, #Gives the ai the available tools 
        tool_choice="auto",
        messages=messages,   
        model = model)
    #Did the ai want to call the tool?
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        #debug confirmation
        if tool_call.function.name == "get_time":
            result = get_time()
            print(f"Tool ran: the time is, {result}")
        elif tool_call.function.name == "get_weather":
            args = json.loads(tool_call.function.arguments)
            result = get_weather(args["city"])
            print(f"Tool ran: Weather report is, {result}")
        elif tool_call.function.name == "play_spotify":
            args = json.loads(tool_call.function.arguments)
            result = play_spotify(args["query"])
        elif tool_call.function.name == "pause_spotify":
            result = pause_spotify()
        elif tool_call.function.name == "next_track":
            result = next_track()
        elif tool_call.function.name == "previous_track":
            result = previous_track()
        

        #add the message to the AIs memory 
        messages.append(response.choices[0].message)
        messages.append({
            "role":"tool",
            "content":result,
            "tool_call_id":tool_call.id    
        })

        final = active_client.chat.completions.create(
            messages=messages,
            model= model,
            tool_choice="none"  # don't call tools again on the final response
        )
        reply = final.choices[0].message.content
    else:
        reply = response.choices[0].message.content
    messages.append({"role":"assistant","content":reply})
    print(f"Jarvis: {reply}\n")
    speak(reply)



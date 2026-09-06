import os
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from datetime import timedelta
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
import cv2
import base64
from ddgs import DDGS



load_dotenv("homeassistant/key.env")#uses python-dotenv library to read the .env file for the api key

api_key = os.environ.get("GROQ_API_KEY")  #assigns the api key to a variable
client = Groq(api_key=api_key)
el_client = ElevenLabs(api_key=os.environ.get("ELEVEN_LABS_KEY"))

kokoro = Kokoro("kokoro-v1.1-zh.onnx", "voices-v1.0.bin") #sets up kokoro 
#tries accessing the groq api link. Will not work without internet access.
try:
    httpx.get("https://api.groq.com", timeout=2)
    ONLINE = True
except:
    ONLINE = False

wakeword_model=Model(wakeword_models=["hey_theo.onnx"], inference_framework="onnx")
import time
def listen_for_wake():
    time.sleep(0.5)
    wakeword_model.reset()
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
        if prediction["hey_theo"] > 0.5:
            stream.stop_stream()
            stream.close()  
            pa.terminate()
            return    # Wake word heard, exit and start recording

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

#Checks internet access to determine whether to use cloud groq model or local ollama model.
def get_client():
    if ONLINE == True:
        return client, "openai/gpt-oss-120b"
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
    except Exception as e:
        print(f"The error is{e}")
        return "Cannot reach weather data, there is no internet available"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
    client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-modify-playback-state user-read-playback-state user-read-recently-played"
))
def get_playback_device():
    """Find a device we can actually play on. Returns a device id, or None."""
    devices = sp.devices().get("devices", [])
    usable = [d for d in devices if not d.get("is_restricted")]
    if not usable:
        return None
    for d in usable:
        if d.get("is_active"):
            return d["id"]
    return usable[0]["id"]      # nothing active — take the first one and wake it
def play_spotify(query):
    try:
        device_id = get_playback_device()
        if device_id is None:
            return "I can't find Spotify running on anything right now. Open it on your phone or computer, then ask me again."

        results = sp.search(q=query, type="track", limit=1)
        if not results["tracks"]["items"]:
            return f"I couldn't find a song called {query}."

        track = results["tracks"]["items"][0]
        sp.start_playback(device_id=device_id, uris=[track["uri"]])
        return f"Playing {track['name']} by {track['artists'][0]['name']}"
    except Exception as e:
        print(f"Spotify error is {e}")
        return "Couldn't play that. Ensure Spotify is open on your device."
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

def wait_for_screen():
    cam = cv2.VideoCapture(0)
    speak("Please hold your screen up to the camera.")
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_area = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if len(approx) == 4 and area > best_area:
                best_area = area
        print(f"Area:{best_area}")
        if best_area > 80000:
            # Screen is large and clear enough
            cam.release()
            _, buffer = cv2.imencode(".jpg", frame)
            return base64.b64encode(buffer).decode("utf-8")
        
        elif best_area > 30000:
            # Screen partially visible — warn the user
            speak("I can see a screen but it's not fully in view. Please move it closer.")
            import time
            time.sleep(3)  # wait before checking again so it doesn't spam
        
        # Below 30000 — nothing detected, keep waiting silently

    cam.release()

def analyze_screen(image_data):
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "This is a screen shown by an older user having a tech issue. What error or problem do you see? Give a simple 1-2 sentence description."
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content
#waits for user to show screen, captures screen, sends the image for analysis
def analyze_error():
    image_data = wait_for_screen() 
    return analyze_screen(image_data)


import threading

def set_timer(time, label):
    def timer_done():
        data, samplerate = sf.read("alarm_sound.wav")
        for i in range(2):
            sd.play(data, samplerate)
            sd.wait()
        speak(f"Time's up! {label}")
    
    t = threading.Timer(time, timer_done)
    t.start()
    return f"{label} set for {time} seconds."
active_timers = {}  # store timers by label

def set_reminder(reminder, hours):
    def reminder_done():
        speak(f"Reminder: {reminder}")
        active_timers.pop(reminder, None)
    
    seconds = hours * 3600
    t = threading.Timer(seconds, reminder_done)
    t.start()
    active_timers[reminder] = t
    return f"I'll remind you about '{reminder}' in {hours} hours."

def cancel_reminder(reminder):
    if reminder in active_timers:
        active_timers[reminder].cancel()
        active_timers.pop(reminder)
        return f"Reminder cancelled."
    return "No reminder found with that name."
            
#Internet access
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        summary = ""
        for r in results:
            summary += f"{r['title']}: {r['body']}\n\n"
        return summary
    except Exception as e:
        print(f"Search error: {e}")
        return "Couldn't search the web right now"




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
            "description":"Gets current weather data in a specified city, including temperature and a description of conditions. This tool requires the user to specify the city they're in. If the user asks 'what's the weather like here?' Use their default location as the city. ",
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
    },
    {
        "type":"function",
        "function": {
            "name":"analyze_error",
            "description":"Captures and analyzes a picture of the error message experienced by the user. This picture will be used to help with troubleshooting. ALWAYS use when the user describes an error message that is vague/ when they mention an error screen popping up. ",
            "parameters":{
                "type":"object",
                "properties":{}
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"set_timer",
            "description":"Accepts a time from the user, and plays an alarm sound when the time is reached.",
            "parameters":{
                "type":"object",
                "properties":{
                    "time":{
                        "type":"integer",
                        "description":"The time the user wishes the alarm to go off by.Duration in seconds. Convert from what the user says e.g. '2 minutes' = 120, '1 hour' = 3600."
                    },
                    "label": {
                        "type":"string",
                        "description":"The purpose of the alarm."
                    }
                    
                },
                "required":["time","label"] #tells the AI it Has to have this info
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"set_reminder",
            "description":"The user gives something you should remind them about, as well as how often to remind them. This tool allows for the user to be constantly reminded about tasks.",
            "parameters":{
                "type":"object",
                "properties":{
                    "reminder":{
                        "type":"string",
                        "description":"The specific task the user wishes to be reminded about."
                    },
                    "hours": {
                        "type":"integer",
                        "description":"The amount of hours that must elapse before the user is reminded about their task again."
                    }
                    
                },
                "required":["reminder","hours"] #tells the AI it Has to have this info
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancels an active reminder the user no longer needs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder": {
                        "type": "string",
                        "description": "The reminder message to cancel."
                    }
                },
                "required": ["reminder"]
            }
        }
    },
     {
        "type":"function",
        "function": {
            "name":"web_search",
            "description":"This tool searches the internet based on information given by the user. ALWAYS use this tool when the user says 'Search for...' or asks about something you don't know about. Also use this if, say, the user asks for your thoughts on a person.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"WHat the user wants to search the internet for"
                    }
                },
                "required":["query"] #tells the AI it Has to have this info
            }
        }
    }

    
]
active_client, model = get_client()
messages=[{"role": "system", #tells the system its role
        "content": """You are a friendly home assistant named THEO, short for Technical Help, Explanations and Optimization, who specializes in 
tech support for older users with little technological experience.
The default location of the user is Kingston, Jamaica.
You have tools available to control Spotify, check weather, and get the time.
IMPORTANT: When you need to use a tool, use it directly — never write out function calls as text.
Always keep responses short and simple — 2 sentences maximum.
Your creator is Aiden Bramwell, an 18 year old student at Campion College who is also an aspiring electrical engineer. If a user asks about him, tell them this.
Avoid technical jargon. If giving instructions, use one step at a time.
Good response example: 'Your printer may be out of paper. Try opening the paper tray and refilling it.'
Bad response example: 'There are several potential causes for this issue. First, let's diagnose whether it's a connectivity problem, a driver issue, or a hardware malfunction...'"""
    }]

CONVERSATION_TIMEOUT = 15  # seconds before returning to wake word mode

while True:
    filename = os.path.dirname(__file__) + "/userspeech.wav"
    
    # Wait for wake word
    listen_for_wake()
    speak("Yes?")
    
    # Keep conversation going until silence
    while True:
        print("Recording for 5 seconds... speak now")
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        write(filename, 16000, audio)
        print("Done recording\n")
        
        usertext = transcribe(filename)
        print(f"You said: {usertext}")
        
        if usertext.strip() in [".", ""]:
            print("Nothing heard — returning to wake word mode\n")
            speak("I'm still here if you need me.")
            break  # exit inner loop, go back to wake word
        
        if "quit" in usertext.lower():
            speak("Goodbye!")
            exit()
        
        messages.append({"role": "user", "content": usertext})
        
        response = None
        for temp in (1.0, 0.6, 0.2):
            try:
                response = active_client.chat.completions.create(
                    tools=tools,
                    tool_choice="auto",
                    messages=messages,
                    model=model,
                    temperature=temp
                )
                break
            except Exception as e:
                print(f"Tool call failed at temperature {temp}: {e}")

        if response is None:
            messages.pop()
            speak("Sorry, I had trouble with that one. Could you say it again?")
            continue

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "get_time":
                result = get_time()
            elif tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                result = get_weather(args["city"])
            elif tool_call.function.name == "play_spotify":
                args = json.loads(tool_call.function.arguments)
                result = play_spotify(args["query"])
            elif tool_call.function.name == "pause_spotify":
                result = pause_spotify()
            elif tool_call.function.name == "next_track":
                result = next_track()
            elif tool_call.function.name == "previous_track":
                result = previous_track()
            elif tool_call.function.name == "analyze_error":
                result = analyze_error()
            elif tool_call.function.name == "set_timer":
                args = json.loads(tool_call.function.arguments)
                result = set_timer(args["time"], args["label"])
            elif tool_call.function.name == "set_reminder":
                args = json.loads(tool_call.function.arguments)
                result = set_reminder(args["reminder"], args["hours"])
            elif tool_call.function.name == "cancel_reminder":
                args = json.loads(tool_call.function.arguments)
                result = cancel_reminder(args["reminder"])
            elif tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = web_search(args["query"])

            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })

            final = active_client.chat.completions.create(
                messages=messages,
                model=model
            )
            reply = final.choices[0].message.content
        else:
            reply = response.choices[0].message.content

        if reply and "<function" in reply:
            print("Model wrote a tool call as text instead of calling it — discarding.")
            messages.pop()
            speak("Sorry, I couldn't do that one. Could you ask me again?")
            continue
        messages.append({"role": "assistant", "content": reply})
        print(f"THEO: {reply}\n")
        speak(reply)
        # Loop continues — user can keep talking without wake word
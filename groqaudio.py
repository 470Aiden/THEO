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



load_dotenv("key.env")#uses python-dotenv library to read the .env file for the api key
api_key = os.environ.get("GROQ_API_KEY")  #assigns the api key to a variable
client = Groq(api_key=api_key)

#Checks internet access to determine whether to use cloud whisper model or local whisper model.

def transcribe(filename):
    try:
        httpx.get("https://api.groq.com", timeout=2)
        # Online — use Groq Whisper
        print("[Online - using Groq Whisper]")
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                language="en"
            )
        return transcription.text
    except:
        # Offline — use local Whisper
        print("[Offline - using local Whisper]")
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(filename, fp16=False)
        return result["text"]
#Checks internet access to determine whether to use cloud groq model or local ollama model.
def get_client():
    try:
        #tries accessing the groq api link. Will not work without internet access.
        httpx.get("https://api.groq.com",timeout=2)
        print("Online - Using Groq")
        return client, "llama-3.1-8b-instant"
        
    except:
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
    }
]
active_client, model = get_client()
messages=[{"role": "system", #tells the system its role
        "content": """You are a friendly home assistant named Jarvis who specializes in 
        tech support for older users with little technological experience.
        Always keep responses short and simple — 2 sentences maximum.
        Avoid technical jargon. If giving instructions, use one step at a time.
        Good response example: 'Your printer may be out of paper. Try opening the paper tray and refilling it.'
        Bad response example: 'There are several potential causes for this issue. First, let's diagnose whether it's a connectivity problem, a driver issue, or a hardware malfunction...'"""
    }]
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
        #add the message to the AIs memory 
        messages.append(response.choices[0].message)
        messages.append({
            "role":"tool",
            "content":result,
            "tool_call_id":tool_call.id    
        })

        final = active_client.chat.completions.create(
            messages=messages,
            model= model
        )
        reply = final.choices[0].message.content
    else:
        reply = response.choices[0].message.content
    messages.append({"role":"assistant","content":reply})
    print(f"Jarvis: {reply}\n")



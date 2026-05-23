import pyaudio as pa
import sounddevice as sd
import soundfile as sf
import datetime

target = datetime.datetime.now() + datetime.timedelta(seconds=10)
while True:
    if datetime.datetime.now() == target:
        print("Remember to take the chicken out the freezer")
        target = datetime.datetime.now() + datetime.timedelta(seconds=10)

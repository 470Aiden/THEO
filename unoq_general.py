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
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import base64
import struct
import threading
import time
import sys
from ddgs import DDGS


from theo_node import TheoNode, NodeUnavailable
from find_node import find_node

# Find the node FIRST, before anything tries to talk to it.
#
# Both addresses it has used are listed, so whichever one it is on
# answers immediately. If neither does, the search works out which
# network you are actually on and looks there.

NODE_IP = find_node(known=("YOUR_NODE_IP_1", "YOUR_NODE_IP_2")) or "YOUR_NODE_IP_1"

print("[THEO] Node address:", NODE_IP)

# THEO's ears and eyes. Nothing happens until you ask it for something.
#
# The first argument is a LIST of full addresses, not one bare address.
# Handing it a bare string is silently wrong rather than loudly wrong:
# tuple("192.168.x.x") splits into single characters, and nothing
# raises to tell you. So: full "http://" addresses, in a tuple.
#
# The address we just found goes first, because it has already answered.
# The name is kept behind it as a second chance in case the address
# changes while THEO is running.

node = TheoNode(hosts=("http://" + NODE_IP, "http://theo-esp32.local"))

print(
    "[THEO] Node reachable:" ,
    "yes" if node.is_ready() else "no -- camera and microphone offline"
)

ENV_FILE = os.path.join(os.path.dirname(__file__), "key.env")

print("Loading:", ENV_FILE)
print("Exists:", os.path.exists(ENV_FILE))

load_dotenv(ENV_FILE)
print("Current directory:", os.getcwd())
print("key.env in use:", ENV_FILE)

print("Spotify ID loaded:",
      bool(os.environ.get("SPOTIFY_CLIENT_ID")))

print("Spotify secret loaded:",
      bool(os.environ.get("SPOTIFY_CLIENT_SECRET")))

# True  -> THEO listens through the node's microphone
# False -> THEO reads typed lines (only useful with a keyboard attached)
USE_NODE_MIC = True
# The vision model used for pictures. It must be one your Groq account has
# access to - a model your account does not have returns 404 on every picture.
VISION_MODEL = "qwen/qwen3.8-27b"


api_key = os.environ.get("GROQ_API_KEY")  #assigns the api key to a variable
client = Groq(api_key=api_key)
el_client = ElevenLabs(api_key=os.environ.get("ELEVEN_LABS_KEY"))

# The offline voice. It is switched OFF: the Kokoro line below is commented
# out, so there is no offline voice until it is switched on. To switch it on,
# uncomment that line and put its two model files next to this program.
kokoro = None
# kokoro = Kokoro("kokoro-v1.1-zh.onnx", "voices-v1.0.bin") #sets up kokoro

# Tries accessing the groq api link. Will not work without internet access.
try:
    httpx.get("https://api.groq.com", timeout=2)
    ONLINE = True
    print("[THEO] Internet connection available")
except Exception:
    ONLINE = False
    print("[THEO] Internet connection unavailable")

# wakeword_model=Model(wakeword_models=["hey_theo.onnx"], inference_framework="onnx")


def listen_for_wake():
    time.sleep(0.5)
    wakeword_model.reset()

    print("Listening for wake word")

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=1280,
        dtype="int16",
        channels=1
    ) as stream:

        while True:
            audio_chunk, overflowed = stream.read(1280)

            if overflowed:
                print("Audio buffer overflow")

            audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)

            prediction = wakeword_model.predict(audio_chunk)

            if prediction["hey_theo"] > 0.5:
                return


local_whisper = WhisperModel("base", device="cpu", compute_type="int8")


#Checks internet access to determine whether to use cloud whisper model or local whisper model.
def transcribe(filename):
    if ONLINE == True:
        # Online - use Groq Whisper
        print("[Online - using Groq Whisper]")
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                language="en"
            )
        return transcription.text
    else:
        # Offline - use local Whisper
        print("[Offline - using local Whisper]")
        segments, _ = local_whisper.transcribe(filename, language="en")
        return " ".join([s.text for s in segments])


# How long THEO listens for one question. The board's hard limit is
# ten seconds -- ask for more and theo_node will tell you it clamped.
LISTEN_MS = 6000


def listen(duration_ms=LISTEN_MS):
    """
    Record a question on the node and turn it into text.

    Returns the transcript, or None if the node could not be reached.
    """

    try:
        # Keep alerts out of the recording, and finish any current alert
        # before announcing that it is the user's turn.
        with AUDIO_LOCK:
            print("THEO: I'm listening.")
            if getattr(listen, "heard_last_time", True):
                try:
                    play_on_node(LISTEN_BEEP, NODE_RATE, timeout=3)
                except Exception as error:
                    print(f"[listen] could not play the ready beep: {error}")

            path = node.record_audio(duration_ms, path="theo_input.wav")

    except NodeUnavailable as error:
        print(f"[node] {error}")
        say(error.spoken)

        # The node is not answering. Without this the program spins,
        # printing the same message once a second forever. Wait a little
        # longer each time it fails, up to half a minute - and reset the
        # moment it works again, so a quiet room is never punished.
        listen.missing_rounds = getattr(listen, "missing_rounds", 0) + 1

        time.sleep(min(listen.missing_rounds * 2, 30))

        listen.heard_last_time = False

        return None

    listen.missing_rounds = 0

    text = transcribe(path).strip()

    # A quiet room does not come back empty. The transcriber turns silence
    # into punctuation - "..." or just "." - and THEO was answering those
    # as though they were questions. If there is not a single letter or
    # number in it, nobody actually spoke.
    if not any(character.isalnum() for character in text):
        listen.heard_last_time = False
        return None

    listen.heard_last_time = True

    return text or None


# ------------------------------------------------------------------
# ALL SOUND GOES TO THE NODE.
#
# The speaker is on the node, in the room. The computer running this
# program has no speaker anyone is listening to. Until 2026-09-28 two
# things still played through the computer's own sound - the alarm chime
# and the offline voice - and the timer test proved it: the words
# "Time's up" reached you, the chime never did.
#
# So there is now one way to make a sound: play_on_node(). Everything
# that should be heard goes through it.
#
# (listen_for_wake, near the top, still reads the computer's microphone.
# That one is rebuilt with the wake word, because the only microphone
# in the room is on the node.)
# ------------------------------------------------------------------

NODE_RATE = 16000      # the node plays 16,000 samples every second
NODE_PIECE = 24000     # the most the node can hold at once
AUDIO_LOCK = threading.RLock()


def to_eight_bit(pcm):
    """Turn 16-bit samples from ElevenLabs into the 8-bit values the node's pin wants."""

    # Each sample is two bytes. If the stream was cut mid-sample there is an
    # odd byte left over - drop it rather than fail. Losing one sample out of
    # thousands is inaudible.
    count = len(pcm) // 2
    samples = np.frombuffer(pcm[:count * 2], dtype="<i2")

    return eight_bit_from_samples(samples.astype(np.float32) / 32768.0)


def eight_bit_from_samples(samples):
    """
    Sound as numbers from -1 to 1  ->  the 256 levels the node's pin can make.

    The old conversion always rounded DOWN, which nudged every sample the
    same way; this rounds to the NEAREST level. It also converts the whole
    clip in one step instead of one sample at a time, so a long reply is
    ready to send sooner.
    """

    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)

    return np.round((samples + 1.0) * 127.5).astype(np.uint8).tobytes()


def resample_for_node(samples, sample_rate):
    """Change any sound to the node's 16,000 samples a second."""

    if sample_rate == NODE_RATE or len(samples) == 0:
        return samples

    # Going DOWN in rate: first remove the sounds too high for the new rate
    # to hold. Left in, they fold back as a harsh edge on the voice.
    if sample_rate > NODE_RATE:
        taps = 63
        centre = np.arange(taps) - (taps - 1) / 2.0
        cutoff = 0.45 * NODE_RATE / float(sample_rate)
        kernel = np.sinc(2.0 * cutoff * centre) * np.hamming(taps)
        kernel = kernel / kernel.sum()
        samples = np.convolve(samples, kernel, mode="same")

    wanted = int(round(len(samples) * NODE_RATE / float(sample_rate)))
    old_times = np.arange(len(samples)) / float(sample_rate)
    new_times = np.arange(wanted) / float(NODE_RATE)

    return np.interp(new_times, old_times, samples).astype(np.float32)


def send_to_node(sound, timeout=30):
    """Send the node 8-bit sound in pieces it can hold, then wait until it has finished playing."""

    # The board can only hold about two and a half seconds of sound at a
    # time. From your own logs: every clip up to 37,895 bytes played, and
    # every clip above that reset the connection. It has no spare memory
    # chip, and the speak handler keeps a second copy of the whole clip
    # while it plays. So the sound goes over in pieces the board can hold.

    if len(sound) == 0:
        return

    with AUDIO_LOCK:
        for start in range(0, len(sound), NODE_PIECE):
            response = requests.post(
                "http://" + NODE_IP + "/audio/speak",
                data=sound[start:start + NODE_PIECE],
                timeout=timeout
            )
            response.raise_for_status()

        # The sketch acknowledges each chunk BEFORE playing it. Its
        # synchronous server can answer this next request only AFTER the
        # last chunk finishes. This also accounts for Wi-Fi delays, which
        # a guess based on total audio duration cannot do.
        response = requests.get("http://" + NODE_IP + "/status", timeout=timeout)
        response.raise_for_status()
        if response.json().get("device") != "THEO-ESP32":
            raise RuntimeError("The speaker address did not answer as a THEO node")


def play_on_node(samples, sample_rate, timeout=30):
    """
    Play any sound through the node's speaker - the one in the room.

    samples: numbers from -1 to 1 (or 16-bit whole numbers), one channel or
    two, at any rate. It is turned into exactly what the node understands.
    """

    samples = np.asarray(samples)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)       # two channels -> one

    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32)

    samples = resample_for_node(samples, sample_rate)

    send_to_node(eight_bit_from_samples(samples), timeout=timeout)


def make_tone(pitches, seconds_each, volume):
    """A short tone for each pitch, one after another, with soft edges so it does not click."""

    count = int(NODE_RATE * seconds_each)
    edge = max(1, min(int(0.010 * NODE_RATE), count // 2))
    times = np.arange(count) / float(NODE_RATE)

    envelope = np.ones(count, dtype=np.float32)
    envelope[:edge] = np.linspace(0.0, 1.0, edge)
    envelope[-edge:] = np.linspace(1.0, 0.0, edge)

    parts = [np.sin(2 * np.pi * pitch * times) * envelope * volume
             for pitch in pitches]

    return np.concatenate(parts).astype(np.float32)


# "Your turn." A quick, soft, rising pair of notes - about a seventh of a second.
LISTEN_BEEP = make_tone((880, 1320), 0.07, 0.35)

# The alarm, when there is no alarm_sound.wav next to this program:
# three rising pairs of notes with a short gap after each.
_ALARM_PAIR = make_tone((988, 1319), 0.15, 0.6)
_ALARM_GAP = np.zeros(int(0.12 * NODE_RATE), dtype=np.float32)
ALARM_TONE = np.concatenate([_ALARM_PAIR, _ALARM_GAP] * 3)


def speak(reply):

    if ONLINE == True:
        audio = el_client.text_to_speech.convert(
            text=reply,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="pcm_16000"
        )

        pcm = b"".join(audio)

        sound = to_eight_bit(pcm)

        # If the voice service gave us nothing back, say so. Without this
        # nothing is sent and you get silence with no clue anywhere - not
        # here, and not on the node, because the node only prints when
        # something actually arrives.
        if len(sound) == 0:
            print("[speak] the voice service returned no audio for that reply")
            return

        # Still ONE voice call, so this costs nothing extra in credits.
        send_to_node(sound)

    else:
        # THE OFFLINE VOICE.
        #
        # Earlier notes said THEO could already speak with no internet. That
        # was wrong twice over: the voice is switched off (see "kokoro = None"
        # near the top), and even switched on it played through the
        # computer's own sound, not the node - so with the internet down,
        # THEO would have gone silent in the room.
        #
        # It now goes to the node like everything else. It still has to be
        # switched on before it can be heard.
        if kokoro is None:
            raise RuntimeError("the offline voice is not switched on yet")

        samples, sample_rate = kokoro.create(
            reply,
            voice="af_heart",
            speed=1.0,
            lang="en-us"
        )

        play_on_node(samples, sample_rate)


def say(text, force=False):
    """
    Say something, and never let the voice layer kill the conversation.

    Always prints. Speaks only when speaking is worth paying for.

    Pass force=True to speak something even if it was just said.

    The audio lock covers generation, every audio chunk, and completion.
    Timer threads cannot enter halfway through another sentence.
    """

    import time

    # How long the same exact sentence must wait before it may be spoken
    # again. 300 seconds = 5 minutes. Raise it to be stricter.
    REPEAT_COOLDOWN = 300

    # How long to stay quiet after the voice service reports you are out
    # of credits. 900 seconds = 15 minutes.
    QUOTA_PAUSE = 900

    with AUDIO_LOCK:
        if not hasattr(say, "spoken_at"):
            say.spoken_at = {}
            say.blocked_until = 0.0

        print(f"THEO: {text}")
        now = time.time()

        if now < say.blocked_until:
            return

        stale = [phrase for phrase, when in say.spoken_at.items()
                 if (now - when) > REPEAT_COOLDOWN]
        for phrase in stale:
            del say.spoken_at[phrase]

        if not force:
            last_said = say.spoken_at.get(text)
            if last_said is not None and (now - last_said) < REPEAT_COOLDOWN:
                return

        # Generation can cost credits even if delivery subsequently fails.
        say.spoken_at[text] = now
        try:
            speak(text)
        except Exception as error:
            detail = str(error)
            lowered = detail.lower()
            if "quota" in lowered or "credits remaining" in lowered:
                say.blocked_until = now + QUOTA_PAUSE
                minutes = int(QUOTA_PAUSE / 60)
                print(f"[voice out of credits - pausing speech for {minutes} minutes]")
            else:
                print(f"[voice unavailable: {detail}]")


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
        return local_client, "llama3.2:3b"


def get_time():
    return datetime.now().strftime("%I:%M %p") #formats time it terms of hour,minute, AM/PM


def get_weather(city):
    try:
        load_dotenv(ENV_FILE)
        weather_key = os.environ.get("WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The current temperature {temp} degrees C, {description} in {city}"
    except Exception as e:
        print(f"The error is {e}")
        return "Cannot reach weather data, there is no internet available"


# SPOTIFY'S LOGIN IS KEPT NEXT TO THIS PROGRAM.
#
# Spotify remembers your login in a file. Left to itself it calls that file
# ".cache" and puts it in whatever folder the program starts in - for you,
# /app. But /app/.cache is already a FOLDER: it holds this program's Python
# install (/app/.cache/.venv). So Spotify could never read a saved login or
# save one, and every request for music started a brand new sign-in.
SPOTIFY_TOKEN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".spotify_token"
)

SPOTIFY_NOT_LINKED = ("Spotify is not linked to me yet. It needs signing in "
                      "once, on a computer with a screen.")


class SpotifyOAuthNoPrompt(SpotifyOAuth):
    """
    Spotify sign-in that NEVER waits for a person on this board.

    Signing in means opening a web page and pressing Agree. This board has
    no screen, so Spotify opened a page nobody could see and waited for it
    forever - and THEO froze with it. That is exactly where the 2026-09-28
    log stopped.

    Signing in now happens once, on a computer, with spotify_login.py. If a
    login is ever needed here, this gives up at once instead of waiting.
    """

    def get_auth_response(self, open_browser=None):
        raise RuntimeError("Spotify is not signed in on this device")


sp = spotipy.Spotify(auth_manager=SpotifyOAuthNoPrompt(
    client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
    client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-modify-playback-state user-read-playback-state user-read-recently-played",
    cache_handler=CacheFileHandler(cache_path=SPOTIFY_TOKEN_FILE),
    open_browser=False
))


def spotify_linked():
    """True only when a saved Spotify login is here. Never starts a sign-in."""
    try:
        return bool(sp.auth_manager.cache_handler.get_cached_token())
    except Exception:
        return False


def spotify_problem(error):
    """A Spotify error, in words a person can act on."""
    text = str(error).lower()
    if "premium" in text:
        return "Spotify only lets me start songs on a Premium account."
    if "not signed in" in text:
        return SPOTIFY_NOT_LINKED
    return "Couldn't play that. Ensure Spotify is open on your device."


def get_playback_device():
    """Find a device we can actually play on. Returns a device id, or None."""
    devices = sp.devices().get("devices", [])
    usable = [d for d in devices if not d.get("is_restricted")]
    if not usable:
        return None
    for d in usable:
        if d.get("is_active"):
            return d["id"]
    return usable[0]["id"]      # nothing active - take the first one and wake it


def play_spotify(query):
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
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
        return spotify_problem(e)


def pause_spotify():
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
    try:
        playback = sp.current_playback()
        if playback and playback["is_playing"]:
            sp.pause_playback()
            return "Track paused"
        else:
            sp.start_playback()
            return "Track resumed"
    except Exception:
        return "Nothing is playing"


def next_track():
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
    try:
        sp.next_track()
        return "Track skipped"
    except Exception:
        return "End of playlist"


def previous_track():
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
    try:
        sp.previous_track()
        return "Now playing the previous track"
    except Exception:
        return "No previous tracks"


def wait_for_screen():
    """
    Ask the person to hold their screen up, then photograph it.

    The node has no live preview, so this asks, waits, and takes one shot --
    which is simpler and, for someone holding up a phone, kinder.
    """

    say("Please hold your screen up to the camera. I'll take the picture in five seconds.")

    time.sleep(5)

    image_bytes = node.capture_image()

    print(f"[node] Captured {len(image_bytes):,} bytes")

    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_screen(image_data):
    """Send the photograph off to be read."""

    response = client.chat.completions.create(
        model=VISION_MODEL,
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
                        "text": (
                            "This is a screen shown by an older user having a "
                            "tech issue. What error or problem do you see? "
                            "Give a simple 1-2 sentence description."
                        )
                    }
                ]
            }
        ],

        # Cap how long the answer may be.
        #
        # Without this the service reserves its DEFAULT answer length - about
        # 1,150 words' worth - and charges your per-minute allowance for all
        # of it, even though the reply above is two sentences. Your allowance
        # is 1,000 a minute, so a single picture was over the line before it
        # even started, which is the "Request too large" you saw.
        #
        # 150 is generous for two sentences and leaves room for about six
        # pictures a minute.

        max_tokens=150
    )

    return response.choices[0].message.content


def analyze_error():
    """Photograph the user's screen and describe what is wrong with it."""

    try:
        image_data = wait_for_screen()

        return analyze_screen(image_data)

    except NodeUnavailable as error:
        print(f"[node] {error}")

        # error.spoken is written to be read out loud.
        return error.spoken


def alarm_file():
    """Find alarm_sound.wav next to this program, wherever it is run from."""

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "alarm_sound.wav")


def play_alarm():
    """
    Sound the alarm through the node, where it can be heard.

    Your own alarm_sound.wav if it is next to this program - played twice,
    as it always was - and a built-in chime if it is not.
    """

    with AUDIO_LOCK:
        path = alarm_file()
        if os.path.exists(path):
            data, rate = sf.read(path, dtype="float32")
            for _ in range(2):
                play_on_node(data, rate)
        else:
            play_on_node(ALARM_TONE, NODE_RATE)


def announce_alert(text):
    """Finish the announcement, then its chime, as one uninterrupted turn."""
    with AUDIO_LOCK:
        say(text, force=True)
        try:
            play_alarm()
        except Exception as error:
            print(f"[alert] could not sound the chime: {error}")


def describe_problem(error):
    """
    Turn a raw error into something a person can act on.

    Returns (what_to_say, worth_trying_again).

    worth_trying_again is the important half. Trying again helps when the
    internet hiccupped. It cannot possibly help when the request itself is
    not allowed - and retrying those just spends your allowance to be told
    no a second time.
    """

    text = str(error)
    low = text.lower()

    # --- permanent. do not try again. -----------------------------------
    if "tool_use_failed" in low or "did not match schema" in low:
        return ("I could not fill in one of my tools correctly for that "
                "request. Trying again would fail the same way.", False)

    if "does not exist or you do not have access" in low:
        return ("The service does not recognise the model name I am set up "
                "to use. That needs changing in the settings.", False)

    if "invalid_api_key" in low or "authentication" in low or "401" in low:
        return ("My key for that service was not accepted. That needs "
                "checking in the settings file.", False)

    if "insufficient" in low and "quota" in low:
        return ("That account is out of credit.", False)

    # --- worth one more try after a pause. ------------------------------
    if "rate_limit" in low or "429" in low or "too large" in low:
        return ("I went over the per-minute allowance. Waiting a few "
                "seconds and trying once more.", True)

    if ("connection" in low or "timeout" in low or "timed out" in low
            or "reset by peer" in low or "temporarily" in low
            or "502" in low or "503" in low or "504" in low):
        return ("The connection to the service dropped. Trying once more.",
                True)

    return (f"Something went wrong talking to the service: {text}", True)


def set_timer(time_seconds, label):
    def timer_done():
        announce_alert(f"Time's up! {label}")

    t = threading.Timer(time_seconds, timer_done)
    t.start()
    return f"{label} set for {time_seconds} seconds."


active_timers = {}  # store timers by label
REMINDER_LOCK = threading.RLock()


def set_reminder(reminder, minutes):
    """
    Remind the user every so many MINUTES, over and over, until cancelled.

    Two things changed here on 2026-09-22.

    The first is why "remind me every two minutes" used to fail outright.
    This tool used to ask for whole HOURS. Two minutes is 0.033 of an hour,
    which is not a whole number, so the request could not be written down at
    all and was refused three times over. Minutes can say everything hours
    could, and everything hours could not.

    The second is that it only ever reminded you ONCE, even though it
    promised to keep reminding you. It now re-arms itself after each
    reminder, so "every two minutes" genuinely means every two minutes.
    Say "cancel the reminder" to stop it.
    """

    minutes = max(1, int(minutes))
    seconds = minutes * 60

    def reminder_done():
        nonlocal timer
        with AUDIO_LOCK:
            with REMINDER_LOCK:
                if active_timers.get(reminder) is not timer:
                    return
            announce_alert(f"Reminder: {reminder}")

        with REMINDER_LOCK:
            if active_timers.get(reminder) is timer:
                timer = threading.Timer(seconds, reminder_done)
                timer.daemon = True
                active_timers[reminder] = timer
                timer.start()

    with REMINDER_LOCK:
        previous = active_timers.pop(reminder, None)
        if previous is not None:
            previous.cancel()
        timer = threading.Timer(seconds, reminder_done)
        timer.daemon = True
        active_timers[reminder] = timer
        timer.start()

    if minutes == 1:
        return f"I'll remind you about '{reminder}' every minute until you tell me to stop."

    return f"I'll remind you about '{reminder}' every {minutes} minutes until you tell me to stop."


def cancel_reminder(reminder):
    with REMINDER_LOCK:
        timer = active_timers.pop(reminder, None)
        if timer is not None:
            timer.cancel()
            return "Reminder cancelled."
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
            summary += f"{r['title']}: {r['body']}  "
        return summary
    except Exception as e:
        print(f"Search error: {e}")
        return "Couldn't search the web right now"


#Describes the tool for the ai, saying "Here is a tool you can use"
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets current weather data in a specified city, including temperature and a description of conditions. This tool requires the user to specify the city they're in. If the user asks 'what's the weather like here?' Use their default location as the city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city the user is in. E.g- London, Paris, etc"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_spotify",
            "description": "Plays a spotify track based on the song and artist name given by the user. Use this whenever the user asks to play a song, and always listen for song and artist names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The name of the song the user wants to play and the artist it is by."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pause_spotify",
            "description": "Pauses/unpauses the spotify track which is currently playing.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "next_track",
            "description": "Plays the next track in the spotify track queue",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "previous_track",
            "description": "Goes back to the previous track in the spotify track queue",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_error",
            "description": "Captures and analyzes a picture of the error message experienced by the user. This picture will be used to help with troubleshooting. ALWAYS use when the user describes an error message that is vague/ when they mention an error screen popping up.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Accepts a time from the user, and plays an alarm sound when the time is reached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "integer",
                        "description": "The time the user wishes the alarm to go off by. Duration in seconds. Convert from what the user says e.g. '2 minutes' = 120, '1 hour' = 3600."
                    },
                    "label": {
                        "type": "string",
                        "description": "The purpose of the alarm."
                    }
                },
                "required": ["time", "label"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "The user gives something you should remind them about, as well as how often to remind them. This tool allows for the user to be constantly reminded about tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder": {
                        "type": "string",
                        "description": "The specific task the user wishes to be reminded about."
                    },
                    "minutes": {
                        "type": "integer",
                        "description": "How many minutes between reminders. Use MINUTES, never hours or fractions of an hour. Two minutes is 2. Half an hour is 30. Three hours is 180. This must be a whole number."
                    }
                },
                "required": ["reminder", "minutes"]
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
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "This tool searches the internet based on information given by the user. ALWAYS use this tool when the user says 'Search for...' or asks about something you don't know about. Also use this if, say, the user asks for your thoughts on a person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What the user wants to search the internet for"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

active_client, model = get_client()

messages = [{
    "role": "system",
    "content": """You are a friendly home assistant named THEO, short for Technical Help, Explanations and Optimization, who specializes in
tech support for older users with little technological experience.
The default location of the user is YOUR_CITY, YOUR_COUNTRY.
You have tools available to control Spotify, check weather, and get the time.
IMPORTANT: When you need to use a tool, use it directly - never write out function calls as text.
Always keep responses short and simple - 2 sentences maximum.
Your creator is YOUR_NAME, YOUR_SHORT_BIO. If a user asks about them, tell them this.
Avoid technical jargon. If giving instructions, use one step at a time.
Good response example: 'Your printer may be out of paper. Try opening the paper tray and refilling it.'
Bad response example: 'There are several potential causes for this issue. First, let's diagnose whether it's a connectivity problem, a driver issue, or a hardware malfunction...'"""
}]

CONVERSATION_TIMEOUT = 15  # seconds before returning to wake word mode

print("")
print("==============================")
print("           THEO")
print("==============================")
print("[THEO] Audio update: 2026-09-28.2")
print("Listening through the node's microphone.")
print("")

while True:

    try:
        if USE_NODE_MIC:
            usertext = listen()

            if not usertext:
                time.sleep(1)
                continue
        else:
            usertext = sys.stdin.readline().strip()

        print(f"You: {usertext}")

        if usertext.lower() == "quit":
            print("THEO: Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": usertext
        })

        response = None

        # TRY AGAIN, BUT ONLY WHEN TRYING AGAIN CAN ACTUALLY WORK.
        #
        # This used to try three times at three different settings, no matter
        # what went wrong. That is right for a dropped connection and wrong
        # for everything else: when the request itself is not allowed, the
        # second and third attempts fail identically and each one still costs
        # you allowance. Your reminder failure on 2026-09-22 was exactly that
        # - three attempts, same refusal, nothing gained.
        #
        # So: a permanent problem stops immediately and tells you why. A
        # passing problem waits five seconds and tries once more.

        last_problem = None

        for attempt, temp in enumerate((1.0, 0.6, 0.2)):
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
                explanation, worth_retrying = describe_problem(e)
                last_problem = explanation
                print(f"[problem] {explanation}")

                if not worth_retrying:
                    break

                is_last = attempt == 2
                if not is_last:
                    print("[problem] waiting 5 seconds before trying again")
                    time.sleep(5)

        if response is None:
            messages.pop()
            say(last_problem or "Sorry, I had trouble with that one.",
                force=True)
            time.sleep(1)
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
                result = set_timer(
                    args["time"],
                    args["label"]
                )

            elif tool_call.function.name == "set_reminder":
                args = json.loads(tool_call.function.arguments)
                result = set_reminder(
                    args["reminder"],
                    args["minutes"]
                )

            elif tool_call.function.name == "cancel_reminder":
                args = json.loads(tool_call.function.arguments)
                result = cancel_reminder(
                    args["reminder"]
                )

            elif tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = web_search(args["query"])

            else:
                result = "I don't know how to use that tool."

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

        messages.append({
            "role": "assistant",
            "content": reply
        })

        say(reply)

    except KeyboardInterrupt:
        print("THEO: Goodbye!")
        break

    except EOFError:
        print("No keyboard attached. Stopping.")
        break

    except Exception as e:
        explanation, _ = describe_problem(e)
        print(f"[problem] {explanation}")
        time.sleep(1)

import os
import collections
import difflib
from scipy.io.wavfile import write
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import requests
import json
import re
import httpx
import queue
from types import SimpleNamespace
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from kokoro_onnx import Kokoro
from elevenlabs.client import ElevenLabs
import elevenlabs.play
try:
    from elevenlabs import VoiceSettings
except ImportError:
    VoiceSettings = None
import soundfile as sf
import numpy as np
from openwakeword.model import Model
from openwakeword.utils import AudioFeatures
from openwakeword.vad import VAD
import inspect
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import base64
import socket
import struct
import threading
import time
import sys
import unicodedata
from ddgs import DDGS


from theo_node import TheoNode, NodeUnavailable, MicStream
from theo_onboarding import PersonalSetup
from find_node import find_node

# Find the node FIRST, before anything tries to talk to it.
#
# The addresses it has used recently are tried first, so whichever one it
# is on answers immediately. If none does, the search works out which
# network you are actually on and looks there. The board has moved more
# than once (.18, .13, .5, .11, .9), so these are only a head start.

BOARD_ADDRESSES = ("192.168.0.9", "192.168.0.11", "192.168.0.13", "192.168.0.18")

# How long THEO waits between looks while the board cannot be found.
BOARD_RETRY = 10.0


def wait_for_board():
    """
    The board's address. If the board is not on the network yet - switched
    off, or still starting up after an upload - THEO waits for it and keeps
    looking, because it can neither hear nor speak without it. (Until
    2026-10-10.5 it carried on with an old fixed address instead, so the
    live microphone and the voice kept trying that address even once the
    board was back, until THEO was restarted.)
    """

    while True:
        address = find_node(known=BOARD_ADDRESSES)

        if address:
            return address

        print("[node] the board is not on the network yet. Is it switched on and "
              f"on the same Wi-Fi? Looking again in {BOARD_RETRY:.0f} seconds.")

        time.sleep(BOARD_RETRY)


NODE_IP = wait_for_board()

print("[THEO] Node address:", NODE_IP)

# THEO's ears and eyes. Nothing happens until you ask it for something.
#
# The first argument is a LIST of full addresses, not one bare address.
# Handing it a bare string is silently wrong rather than loudly wrong:
# tuple("192.168.0.13") splits into twelve single characters, and nothing
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
# Aiden corrected this himself on 2026-09-21 - 3.6 did not exist on his
# account and returned 404 on every picture. 3.8 works; the first thing it
# ever read was a Windows stop code off a photo of his screen.
VISION_MODEL = "qwen/qwen3.8-27b"

# The model that answers on Groq. Without Groq, Ollama answers instead (see
# THE OFFLINE BRAIN).
ONLINE_MODEL = "openai/gpt-oss-120b"


api_key = os.environ.get("GROQ_API_KEY")  #assigns the api key to a variable
client = Groq(api_key=api_key)
el_client = ElevenLabs(api_key=os.environ.get("ELEVEN_LABS_KEY"))

# THE OFFLINE VOICE.
#
# THEO speaks with it whenever ElevenLabs turns a request down or cannot be
# reached, and whenever there is no internet. Two voices work, each from
# files next to this program.
#
# PIPER, the light one: about a tenth of Kokoro's work for each second of
# speech, so on the Uno Q a reply takes seconds rather than most of a
# minute. It sounds a little flatter than Kokoro. It needs
# "pip install piper-tts" and one voice's two files, from
# huggingface.co/rhasspy/piper-voices, for example
#
#     en_US-lessac-medium.onnx  and  en_US-lessac-medium.onnx.json
#
# KOKORO, the more natural one, from the "model-files-v1.0" release of
# kokoro-onnx on GitHub:
#
#     kokoro-v1.0.int8.onnx   88 MB, the voice model
#     voices-v1.0.bin         27 MB, the voices (af_heart is one of them)
#
# The full-size kokoro-v1.0.onnx (310 MB) also works, a little clearer and
# slower. Until 2026-10-10.7 the line here named kokoro-v1.1-zh.onnx, which
# is the Chinese model, beside the English voices file.
#
# With a Piper voice here THEO uses it, and Kokoro otherwise. OFFLINE_VOICE
# chooses instead: "kokoro" or "piper". Either way THEO speaks a sentence at
# a time, making the next sentence while the last one plays (see speak).

OFFLINE_VOICE = ""
KOKORO_MODELS = ("kokoro-v1.0.int8.onnx", "kokoro-v1.0.fp16.onnx", "kokoro-v1.0.onnx")
KOKORO_VOICES = "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"


def piper_voices(here):
    """The Piper voices next to this program: each .onnx file with its .onnx.json beside it."""

    try:
        names = sorted(os.listdir(here))
    except OSError:
        return []

    return [name for name in names if name.endswith(".onnx") and name + ".json" in names]


def start_piper(path):
    """A Piper voice that speaks the way Kokoro does here (see speak), or None if it cannot start."""

    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError:
        print(f"[voice] Found the Piper voice {os.path.basename(path)}, but Piper is not "
              "installed. Run 'pip install piper-tts' to use it.")
        return None

    try:
        piper = PiperVoice.load(path)
    except Exception as error:
        print(f"[voice] Offline voice: Piper could not start ({error}).")
        return None

    def create(text, voice=None, speed=1.0, lang=None):
        # Piper's length_scale is how long each sound lasts: slower is longer.
        chunks = list(piper.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0 / speed)))

        if not chunks:
            return np.zeros(0, dtype=np.float32), piper.config.sample_rate

        return (np.concatenate([chunk.audio_float_array for chunk in chunks]),
                chunks[0].sample_rate)

    print(f"[voice] Offline voice: ready (Piper, {os.path.basename(path)}).")
    return SimpleNamespace(create=create, name="Piper")


def start_offline_voice():
    """The offline voice, ready to speak, or None when its files are not here."""

    here = os.path.dirname(os.path.abspath(__file__))
    wanted = OFFLINE_VOICE.strip().lower()

    if wanted != "kokoro":
        found = piper_voices(here)

        if found:
            voice = start_piper(os.path.join(here, found[0]))

            if voice is not None or wanted == "piper":
                return voice
        elif wanted == "piper":
            print("[voice] Offline voice: off. OFFLINE_VOICE asks for Piper, but there is no "
                  "Piper voice next to this program (a .onnx file with its .onnx.json).")
            return None

    voices = os.path.join(here, KOKORO_VOICES)
    models = [os.path.join(here, name) for name in KOKORO_MODELS
              if os.path.exists(os.path.join(here, name))]

    if not models or not os.path.exists(voices):
        print("[voice] Offline voice: off. To switch it on, put kokoro-v1.0.int8.onnx "
              "and voices-v1.0.bin next to this program, or a Piper voice (see THE "
              "OFFLINE VOICE).")
        return None

    try:
        voice = Kokoro(models[0], voices)
    except Exception as error:
        print(f"[voice] Offline voice: could not start ({error}).")
        return None

    print(f"[voice] Offline voice: ready ({os.path.basename(models[0])}).")
    return voice


offline_voice = start_offline_voice()

# Tries accessing the groq api link. Will not work without internet access.
try:
    httpx.get("https://api.groq.com", timeout=2)
    ONLINE = True
    print("[THEO] Internet connection available")
except Exception:
    ONLINE = False
    print("[THEO] Internet connection unavailable")

# When Groq fails - listening or answering - THEO uses local Whisper and
# Ollama instead, and asks Groq again this many seconds later. With no
# internet at start-up it begins offline and keeps asking, so it goes back
# online by itself once the internet returns.
GROQ_RETRY_AFTER = 60.0
groq_down_until = 0.0 if ONLINE else time.monotonic() + GROQ_RETRY_AFTER

# How long THEO waits on Groq before using the offline version: plenty for
# a normal answer, not long enough to leave someone waiting on a dead line.
GROQ_TIMEOUT = 20.0

# True while THEO cannot reach Groq, so it listens and thinks offline. The
# light ring shows its offline colour (blue) meanwhile, for anyone who
# can't hear THEO say so.
brain_offline = not ONLINE


def brain_lost():
    """Groq could not be reached: from now until it answers again, THEO runs offline."""

    global brain_offline

    if not brain_offline:
        print("[offline] THEO is running offline; the ring shows blue until Groq answers again.")

    brain_offline = True
    show_offline(True)


def brain_reached():
    """Groq answered, so THEO is online."""

    global brain_offline

    if brain_offline:
        print("[offline] Groq answers again; THEO is back online.")

    brain_offline = False
    show_offline(False)


def show_offline(offline):
    """Tell the ring whether THEO is offline. (A theo_node.py older than 2026-10-16.1 cannot.)"""

    if hasattr(node, "set_offline"):
        node.set_offline(offline)


def show_ring(state):
    """Like node.set_ring, but a state an older theo_node.py does not know shows as idle."""

    try:
        node.set_ring(state)
    except ValueError:
        node.set_ring("idle")


# Local Whisper listens when Groq cannot. It is loaded the first time it is
# needed, so it takes no memory while THEO is online; starting without the
# internet, THEO loads it straight away.
WHISPER_MODEL = "base"
local_whisper = None


def offline_listener():
    """Local Whisper, loaded the first time THEO has to listen without Groq."""

    global local_whisper

    if local_whisper is None:
        print(f"[listen] loading local Whisper ({WHISPER_MODEL}) to listen offline...")
        local_whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    return local_whisper


if not ONLINE:
    offline_listener()


# ------------------------------------------------------------------
# NAMES AND PLACES THEO SHOULD HEAR RIGHT
#
# Both speech-to-text services take a short hint: words to expect. With
# it, "Papine" is not written as "a pain", and a granddaughter's name
# comes out spelt the way the family spells it.
#
# Put the names of the person THEO helps and their family in
# FAMILY_NAMES, for example ["Keisha", "Andre", "Miss Joyce"]. Names in
# what THEO remembers about the person are added by themselves (see WHAT
# THEO REMEMBERS). The hint has room for about 100 names and places, so
# keep it to the ones that matter; names THEO keeps getting wrong are the
# best ones to add.
# ------------------------------------------------------------------

FAMILY_NAMES = []

LOCAL_PLACES = [
    "Half Way Tree", "Papine", "Liguanea", "New Kingston", "Cross Roads",
    "Constant Spring", "Barbican", "Mona", "Stony Hill", "Portmore",
    "Spanish Town", "Old Harbour", "May Pen", "Mandeville", "Ocho Rios",
    "Montego Bay", "Negril", "Port Antonio", "Morant Bay", "Linstead",
]

# Groq's hint may be at most 224 tokens; this many letters stays well inside.
HINT_LONGEST = 600


def listening_hint():
    """The hint for both transcribers: "Theo, Keisha, Half Way Tree, Papine." and so on."""

    words = []

    for word in ["Theo"] + list(FAMILY_NAMES) + remembered_names() + list(LOCAL_PLACES):
        word = " ".join(str(word).split())

        if word and word.casefold() not in (w.casefold() for w in words):
            words.append(word)

    hint = ""

    for word in words:
        longer = (hint + ", " + word) if hint else word

        if len(longer) + 1 > HINT_LONGEST:
            break

        hint = longer

    return hint + "."


def speech_segments(filename):
    audio, rate = sf.read(filename, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("The node recording must be mono audio at 16000 Hz")
    if len(audio) == 0:
        return []
    return get_speech_timestamps(audio, vad_options=VadOptions(
        threshold=0.6, min_speech_duration_ms=160,
        min_silence_duration_ms=300, speech_pad_ms=150))


def confident_transcript(segments):
    accepted = []
    for segment in segments:
        if (segment["no_speech_prob"] >= 0.6 or
                segment["avg_logprob"] < -1.0 or
                segment["compression_ratio"] > 2.4):
            continue
        accepted.append(segment["text"].strip())
    return " ".join(accepted)


def transcribe(filename):
    global groq_down_until

    if not speech_segments(filename):
        print("[listen] no speech detected")
        return ""

    hint = listening_hint()

    if time.monotonic() >= groq_down_until:
        # Online - use Groq Whisper, and local Whisper if Groq fails.
        print("[Online - using Groq Whisper]")
        try:
            with open(filename, "rb") as file:
                transcription = client.with_options(
                    max_retries=0, timeout=GROQ_TIMEOUT).audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3-turbo",
                    language="en",
                    prompt=hint,
                    response_format="verbose_json",
                    temperature=0.0
                )
            brain_reached()
            return confident_transcript(transcription.segments)
        except Exception as error:
            print(f"[listen] Groq could not transcribe that ({error}). Using local Whisper.")
            if describe_problem(error)[1]:
                groq_down_until = time.monotonic() + GROQ_RETRY_AFTER
                brain_lost()

    # Offline - use local Whisper. beam_size=1 takes the likeliest words
    # rather than weighing five guesses at once: much quicker on the Uno Q,
    # and for short requests the words almost always come out the same.
    print("[Offline - using local Whisper]")
    started = time.monotonic()
    segments, _ = offline_listener().transcribe(
        filename, language="en", condition_on_previous_text=False,
        initial_prompt=hint, beam_size=1,
        vad_filter=True, vad_parameters={"threshold": 0.6,
                                         "min_speech_duration_ms": 160})
    segments = list(segments)
    print(f"[listen] local Whisper took {time.monotonic() - started:.1f} s")
    return confident_transcript([{
        "text": s.text, "no_speech_prob": s.no_speech_prob,
        "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio
    } for s in segments])


# ------------------------------------------------------------------
# "HEY THEO"
#
# The board streams its microphone to this program all the time, and your
# wake word model - hey_theo.onnx, next to this program - listens to that
# stream right here on the Uno Q. Nothing is recorded, sent anywhere or
# transcribed until it hears "Hey Theo".
#
# Then the ring lights up, THEO chimes, and it takes your request - all
# of it, even when you say it in the same breath as "Hey Theo" - and
# stops when you stop talking. After answering, it keeps listening for
# CONVERSATION_TIMEOUT seconds, so a follow-up needs no "Hey Theo". (Not
# while a song plays: the song's words would count as a follow-up.)
#
# THEO's own voice never wakes THEO. The listener ignores the microphone
# while THEO is talking, and for a moment after.
# ------------------------------------------------------------------

THEO_DIR = os.path.dirname(os.path.abspath(__file__))

WAKE_MODEL_FILE = os.path.join(THEO_DIR, "hey_theo.onnx")

# How sure the model must be, 0 to 1, before THEO wakes. Raise it (0.5,
# 0.6) if THEO wakes when nobody said "Hey Theo"; lower it if it misses
# you. Near misses are printed with their score, to help you pick. 0.4 is
# the value Aiden chose on 2026-10-10.
WAKE_THRESHOLD = 0.4

# openWakeWord needs three helper files beside your model: two that turn
# sound into what the model reads, and Silero's detector for when someone
# is talking. Fetched once, into this program's folder.
WAKE_HELPERS = ("melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx")
WAKE_HELPER_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"

PIECE = 1280                     # 80 ms of sound: what the model reads at a time
PIECE_SECONDS = PIECE / 16000.0

REQUEST_WAIT = 5.0               # after "Hey Theo", how long it waits for you to start
REQUEST_FINISHED = 1.0           # this long without speech means you have finished
REQUEST_LONGEST = 12.0           # the longest request it takes in

SPEECH_LIKELY = 0.5              # the speech detector's score for "someone is talking"
SPEECH_UNLIKELY = 0.35           # ... and for "they have stopped"

# Set while anything plays through THEO's speaker - see send_to_node() -
# so the listener can ignore THEO's own voice, and for TALK_ECHO seconds
# after it ends.
THEO_TALKING = threading.Event()
theo_quiet_since = 0.0
TALK_ECHO = 0.6

# Set while THEO is taking down a request, so a reminder that comes due
# waits for the person to finish (at most ALERT_PATIENCE seconds) instead
# of talking over them. ALERT_PLAYING covers a reminder from its first word
# to the end of its chime, gap included.
HEARING = threading.Event()
ALERT_PLAYING = threading.Event()
ALERT_PATIENCE = 30.0

# Set while THEO waits for "Hey Theo" with nothing else going on, so a
# reminder knows it may switch the ring to its amber pulse straight away.
WAITING_FOR_WAKE = threading.Event()

# Set after a reminder or timer goes off, until the person next speaks to
# THEO: while it is set, the ring pulses amber slowly instead of going dark.
REMINDER_LIGHT = threading.Event()

# Set when a medicine reminder has asked "Did you take it?", so the
# listener takes the answer without "Hey Theo".
ANSWER_WANTED = threading.Event()

# ------------------------------------------------------------------
# THE TALK BUTTON (board update 2026-10-16.1: a button from GPIO 13 to GND)
#
# Press it to talk without "Hey Theo": THEO chimes and listens, as if it
# had heard its name. Press it once THEO has heard you - while it works
# out your words, works out its answer, or talks - and THEO stops at once
# and goes back to waiting. (A press while it is still listening changes
# nothing.) The board sends each press along with the live microphone, so
# nothing extra is asked of the network, and it cuts THEO's voice off by
# itself the moment the button goes down.
# ------------------------------------------------------------------

talk_button = None          # set at start-up, once the live microphone is open
PRESSED = "pressed"         # what Ears.follow_up returns when the button is pressed

# The press count when the request THEO is answering was heard (set by
# hear_request the moment the listening ends). A press after that - while
# THEO works out the words, thinks or talks - stops the answer. None
# between turns.
answering_from = None

# Set when the talk button stops THEO partway through speaking, so the rest
# of that turn (a reminder's chime, the follow-up window) is skipped as well.
# Cleared each time THEO starts to say something new.
STOPPED_BY_BUTTON = threading.Event()


class TalkButton:
    """The talk button on the board, as seen through the live microphone."""

    def __init__(self, mic):
        self.mic = mic
        self.handled = mic.presses

    def presses(self):
        """How many presses have arrived since THEO started."""

        return self.mic.presses

    def take(self):
        """True once for each new press. Presses close together count as one."""

        presses = self.mic.presses

        if presses == self.handled:
            return False

        self.handled = presses

        return True

    def mark(self):
        """Use up any press so far and return the count: only a press after this is new."""

        self.handled = self.mic.presses

        return self.handled

    def board_count(self):
        """The board's own count of presses: THEO's voice carries it, see send_to_node."""

        return self.mic.board_presses


def fetch_wake_helpers():
    """Put openWakeWord's three helper files next to this program, once."""

    for name in WAKE_HELPERS:
        path = os.path.join(THEO_DIR, name)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue

        print(f"[wake] fetching {name} (only the first time)...")

        response = requests.get(WAKE_HELPER_URL + name, timeout=60)
        response.raise_for_status()

        # Written beside it, then renamed, so a cut-off download is never
        # mistaken for the real file next time.
        part = path + ".part"

        with open(part, "wb") as handle:
            handle.write(response.content)

        os.replace(part, path)


class Ears:
    """
    THEO's listening: the live microphone, the wake word model and the
    speech detector, together.
    """

    def __init__(self, mic, wake_model, wake_key, vad):
        self.mic = mic
        self.wake_model = wake_model
        self.wake_key = wake_key
        self.vad = vad
        self.missing_since = None
        self.last_near_miss = 0.0

        # What ended the last wait_for_wake(): "wake" (Hey Theo), "button"
        # (the talk button) or "answer" (a reminder waiting for an answer).
        self.woken_by = "wake"

    def piece(self, timeout=0.5):
        """The next 80 ms of sound, or None. Speaks up if the microphone stays away."""

        piece = self.mic.read(timeout=timeout)

        if piece is not None:
            self.missing_since = None
            return piece

        if not self.mic.connected:
            now = time.monotonic()

            if self.missing_since is None:
                self.missing_since = now
                print(f"[mic] waiting for the board's live microphone at {self.mic.host}: "
                      f"{self.mic.problem or 'connecting'}")

            elif now - self.missing_since > 15:
                self.missing_since = now
                say("I can't hear through my microphone right now. "
                    "Please check that my board has power.")

        return None

    def theo_just_talked(self):
        return (THEO_TALKING.is_set() or ALERT_PLAYING.is_set() or
                (time.monotonic() - theo_quiet_since) < TALK_ECHO)

    def speech_score(self, piece):
        """0 to 1: how sure the detector is that someone is talking in this piece."""

        return float(self.vad.predict(piece, frame_size=640))

    def wait_for_wake(self):
        """
        Wait for "Hey Theo". Returns the last second of sound, which may
        hold the start of your request - or None if THEO is stopping. The
        model can be sure a moment after "Theo" ends, by which time you
        may already be asking; whatever "Hey Theo" this keeps is trimmed
        off the words afterwards.
        """

        recent = collections.deque(maxlen=12)
        self.mic.drain()
        self.wake_model.reset()
        muted = False

        while not STOPPING.is_set():
            # The talk button works like "Hey Theo", with nothing said yet.
            if talk_button is not None and talk_button.take():
                print("[button] talk button pressed: listening")
                self.woken_by = "button"
                return []

            # A medicine reminder has just asked "Did you take it?". (If
            # THEO was busy for a while after asking, the moment has gone:
            # the reminder asks again later.)
            if ANSWER_WANTED.is_set() and not self.theo_just_talked():
                ANSWER_WANTED.clear()

                if time.monotonic() - answer_asked_at <= ANSWER_FRESH:
                    self.woken_by = "answer"
                    return []

                medicine_not_answered()

            piece = self.piece()

            if piece is None:
                continue

            if self.theo_just_talked():
                muted = True
                recent.clear()
                continue

            if muted:
                # THEO's own voice may still be in the model's memory.
                self.wake_model.reset()
                muted = False

            recent.append(piece)

            score = float(self.wake_model.predict(piece).get(self.wake_key, 0.0))

            if score >= WAKE_THRESHOLD:
                print(f"[wake] Hey Theo ({score:.2f})")
                self.wake_model.reset()
                self.woken_by = "wake"
                return list(recent)

            now = time.monotonic()

            if score >= WAKE_THRESHOLD * 0.6 and now - self.last_near_miss > 2.0:
                self.last_near_miss = now
                print(f"[wake] close, but not sure enough ({score:.2f})")

        return None

    def capture(self, before=(), started=False, wait=REQUEST_WAIT):
        """
        Collect what is said from the live microphone until the speaker has
        finished. `before` is sound already heard, kept so the first words
        are never lost. Returns the sound, or None if nobody started
        talking within `wait` seconds.

        While this runs, a reminder that comes due waits (see
        announce_alert). If one starts anyway, its sound is left out.
        """

        HEARING.set()

        try:
            return self._capture(before, started, wait)
        finally:
            HEARING.clear()

    def _capture(self, before, started, wait):
        self.vad.reset_states()

        pieces = list(before)

        # Let the detector hear what came before, so it is not starting cold.
        for piece in pieces:
            self.speech_score(piece)

        heard_speech = started
        speech_run = 0
        quiet_run = 0
        start = time.monotonic()

        while not STOPPING.is_set():
            now = time.monotonic()

            if reset_pending():
                return None

            if not heard_speech and now - start >= wait:
                return None

            if now - start >= wait + REQUEST_LONGEST:
                break

            # THEO is already listening: a press now changes nothing.
            if talk_button is not None:
                talk_button.take()

            piece = self.piece()

            if piece is None:
                continue

            if ALERT_PLAYING.is_set():
                # A reminder began after all. Its words are not yours.
                continue

            pieces.append(piece)
            score = self.speech_score(piece)

            if not heard_speech:
                # A quarter of a second of speech, so the tail end of "Theo"
                # or a cough is not taken for the start of a request.
                speech_run = speech_run + 1 if score >= SPEECH_LIKELY else 0

                if speech_run >= 3:
                    heard_speech = True

            else:
                quiet_run = quiet_run + 1 if score < SPEECH_UNLIKELY else 0

                if quiet_run * PIECE_SECONDS >= REQUEST_FINISHED:
                    break

            if len(pieces) * PIECE_SECONDS >= REQUEST_LONGEST:
                break

        if not heard_speech or not pieces:
            return None

        return np.concatenate(pieces)

    def follow_up(self, seconds):
        """
        Right after THEO answers: wait up to `seconds` for you to say more,
        with no "Hey Theo" needed. Returns the sound, or None - or PRESSED
        if the talk button was pressed, so the caller can chime first.
        """

        self.mic.drain()
        self.vad.reset_states()

        recent = collections.deque(maxlen=4)
        speech_run = 0
        deadline = time.monotonic() + seconds

        while not STOPPING.is_set():
            if talk_button is not None and talk_button.take():
                print("[button] talk button pressed: listening")
                return PRESSED

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                return None

            piece = self.piece(timeout=min(0.5, remaining))

            if piece is None:
                continue

            if self.theo_just_talked():
                recent.clear()
                speech_run = 0
                continue

            recent.append(piece)

            speech_run = speech_run + 1 if self.speech_score(piece) >= SPEECH_LIKELY else 0

            if speech_run >= 3:
                return self.capture(before=list(recent), started=True)

        return None


def load_wake_model():
    """
    Your hey_theo model, inside openWakeWord, with the helper files next to
    this program.

    Two generations of openWakeWord are in use, and they name these
    settings differently. The Uno Q has 0.4: from 0.5 on, openWakeWord
    needs tflite-runtime, which has no build for the Uno Q's Python 3.13,
    so pip installs 0.4 there. A computer with an older Python usually has
    0.5 or later. The feature reader's own settings say which one this is,
    and the model is loaded the way that version expects. Both give the
    same scores with the same helper files.
    """

    melspec = os.path.join(THEO_DIR, "melspectrogram.onnx")
    embedding = os.path.join(THEO_DIR, "embedding_model.onnx")

    settings = inspect.signature(AudioFeatures.__init__).parameters

    if "melspec_model_path" in settings:
        # openWakeWord 0.5 and later
        return Model([WAKE_MODEL_FILE], inference_framework="onnx",
                     melspec_model_path=melspec, embedding_model_path=embedding)

    # openWakeWord 0.4
    return Model([WAKE_MODEL_FILE],
                 melspec_onnx_model_path=melspec, embedding_onnx_model_path=embedding)


def start_ears():
    """Load your wake word and the speech detector, and open the live microphone."""

    if not os.path.exists(WAKE_MODEL_FILE):
        print(f"[wake] hey_theo.onnx is not in {THEO_DIR}. Put it next to main.py.")
        say("I can't find my wake word file. Please put hey_theo dot onnx next to my program.",
            force=True)
        raise SystemExit(1)

    fetch_wake_helpers()

    wake_model = load_wake_model()

    # The model reports its score under its file name: "hey_theo".
    wake_key = os.path.splitext(os.path.basename(WAKE_MODEL_FILE))[0]

    vad = VAD(model_path=os.path.join(THEO_DIR, "silero_vad.onnx"))

    status = node_audio_status() or {}
    version = str(status.get("version", "unknown"))

    if version == "unknown" or version < "2026-10-10.1":
        print(f"[THEO] The board is on update {version}. \"Hey Theo\" needs the "
              "2026-10-10.1 board update - THEO cannot hear until it is flashed.")
    elif version < "2026-10-16.1":
        print(f"[THEO] The board is on update {version}. The talk button and the reminder "
              "and offline lights need the 2026-10-16.1 board update.")
    elif version < "2026-10-31.1":
        print(f"[THEO] The board is on update {version}. Pictures taken the moment THEO "
              "asks, not earlier ones, need the 2026-10-31.1 board update.")

    mic = MicStream(NODE_IP, piece=PIECE).start()

    print(f'[wake] listening for "Hey Theo" (threshold {WAKE_THRESHOLD})')

    return Ears(mic, wake_model, wake_key, vad)


# ------------------------------------------------------------------
# FOLLOWING THE BOARD
#
# The board can restart while THEO is running - after an upload, a power
# blip, Wi-Fi dropping out - and come back at a new address. So whenever
# the live microphone has been away for BOARD_LOOK_AFTER seconds, THEO
# looks for the board again, then less and less often (at most every
# BOARD_LOOK_LONGEST seconds) until the microphone is back. If the board
# answers somewhere new, everything that talks to it moves there.
# ------------------------------------------------------------------

BOARD_LOOK_AFTER = 8.0
BOARD_LOOK_LONGEST = 60.0


def point_at_board(address, mic=None):
    """From now on, talk to the board at `address`: speaker, camera, ring and microphone."""

    global NODE_IP

    NODE_IP = address

    node.hosts = ("http://" + address, "http://theo-esp32.local")

    # TheoNode keeps the address that answered last. Forget it, so its next
    # request goes to the new one.
    node._working_host = None

    if mic is not None:
        mic.host = address


def follow_board(mic):
    """
    Runs beside THEO for as long as it runs (started near the end of this
    program). Does nothing while the live microphone is connected.
    """

    away_since = None
    next_look = 0.0
    gap = BOARD_RETRY
    answered_here = 0

    while not STOPPING.is_set():
        time.sleep(1.0)

        if mic.connected:
            away_since = None
            gap = BOARD_RETRY
            answered_here = 0
            continue

        now = time.monotonic()

        if away_since is None:
            away_since = now
            next_look = now + BOARD_LOOK_AFTER

        if now < next_look:
            continue

        address = find_node(known=(NODE_IP,) + BOARD_ADDRESSES)

        if address and address != NODE_IP:
            print(f"[node] the board is now at {address} (it was at {NODE_IP}); following it")
            point_at_board(address, mic)
            answered_here = 0

        elif address:
            # Just after a restart the board answers a moment before the
            # microphone reconnects, so only say so the second time.
            answered_here += 1

            if answered_here == 2:
                print(f"[node] the board answers at {address}, but its live "
                      "microphone does not. If this lasts, restart the board.")

        else:
            print("[node] the board is not on the network. Is it switched on? "
                  f"Looking again in {gap:.0f} seconds.")

        next_look = time.monotonic() + gap
        gap = min(gap * 2, BOARD_LOOK_LONGEST)


def chime():
    """"Your turn." The quick rising pair of notes."""

    try:
        play_on_node(LISTEN_BEEP, NODE_RATE, timeout=3)
    except Exception as error:
        print(f"[listen] could not play the chime: {error}")


def transcribe_sound(sound):
    """Turn the sound into text. Returns "" if there were no words in it."""

    path = os.path.join(THEO_DIR, "theo_input.wav")

    sf.write(path, sound, 16000, subtype="PCM_16")

    text = transcribe(path).strip()

    # A quiet room does not come back empty. The transcriber turns silence
    # into punctuation - "..." or just "." - and THEO was answering those
    # as though they were questions. If there is not a single letter or
    # number in it, nobody actually spoke.
    if not any(character.isalnum() for character in text):
        return ""

    return text


def show_waiting_ring():
    """
    Dark while THEO waits for "Hey Theo" - a slow amber pulse while a
    reminder waits to be seen, or a soft white glow while its eyes are on.
    """

    if REMINDER_LIGHT.is_set():
        show_ring("reminder")
    elif eyes_are_on():
        show_ring("looking")
    else:
        show_ring("idle")


def hear_request(follow_up_seconds=0.0):
    """
    The next thing said to THEO, as text - or None.

    With follow_up_seconds, THEO has just answered: it listens that long
    for more, no "Hey Theo" needed. Otherwise it waits for "Hey Theo".
    """

    global answering_from

    if follow_up_seconds > 0:
        node.set_ring("listening")

        sound = ears.follow_up(follow_up_seconds)

        if reset_pending():
            return None

        if sound is None:
            return None

        if isinstance(sound, str):
            # The talk button, in the middle of the follow-up window.
            chime()

            sound = ears.capture()

            if sound is None:
                print("[listen] nothing was said after the talk button")
                return None

    else:
        show_waiting_ring()

        WAITING_FOR_WAKE.set()

        try:
            before = ears.wait_for_wake()
        finally:
            WAITING_FOR_WAKE.clear()

        if reset_pending():
            return None

        if before is None:
            return None

        node.set_ring("listening")

        # A song playing? Pause it, so THEO can hear you and you can hear
        # THEO. Your words over the song so far are still in `before`.
        if music_on_speaker(node_audio_status()):
            music_turn.begin()

        if ears.woken_by == "answer":
            # A medicine reminder asked "Did you take it?". The answer
            # needs no "Hey Theo" and no chime.
            ANSWER_WANTED.clear()

            sound = ears.follow_up(ANSWER_SECONDS)

            if isinstance(sound, str):
                chime()
                sound = ears.capture()

            if sound is None:
                print("[reminder] nobody answered the reminder's question")
                medicine_not_answered()

                if music_turn.holds_paused_song():
                    music_turn.finish()

                return None

        else:
            chime()

            sound = ears.capture(before=before)

            if sound is None:
                if ears.woken_by == "button":
                    print("[listen] nothing was said after the talk button")
                else:
                    print('[listen] nothing followed "Hey Theo"')

                if music_turn.holds_paused_song():
                    music_turn.finish()

                return None

    # From here on THEO is working on what you said, so a press stops it,
    # even while the words are still being worked out, which can take a few
    # seconds offline. (Until 2026-10-16.3 such a press waited, and after
    # the answer it started a new turn instead.)
    if reset_pending():
        return None

    if talk_button is not None:
        answering_from = talk_button.mark()

    node.set_ring("thinking")

    text = transcribe_sound(sound)

    # "Hey Theo" itself may be at the start of what was heard.
    rest = after_name(text) if text else None

    if rest is not None:
        text = rest

    if not text:
        print("[listen] no words in that")

        if music_turn.holds_paused_song() and not music_turn.waiting_for_answer:
            music_turn.finish()

        return None

    return text


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
# that should be heard goes through it - which is also how the "Hey Theo"
# listener knows when to ignore THEO's own voice (see send_to_node).
# ------------------------------------------------------------------

NODE_RATE = 16000      # the node plays 16,000 samples every second
NODE_PIECE = 4096      # leave memory for the camera, microphone and HTTP body copy
AUDIO_LOCK = threading.RLock()
STOPPING = threading.Event()


def to_eight_bit(pcm, loudness=1.0):
    """Turn 16-bit samples from ElevenLabs into the 8-bit values the node's pin wants."""

    # Each sample is two bytes. If the stream was cut mid-sample there is an
    # odd byte left over - drop it rather than fail. Losing one sample out of
    # thousands is inaudible.
    count = len(pcm) // 2
    samples = np.frombuffer(pcm[:count * 2], dtype="<i2").astype(np.float32) / 32768.0

    if loudness != 1.0:
        samples = louder(samples, loudness)

    return eight_bit_from_samples(samples)


def eight_bit_from_samples(samples):
    """
    Sound as numbers from -1 to 1  ->  the 256 levels the node's pin can make.

    The old conversion always rounded DOWN, which nudged every sample the
    same way; this rounds to the NEAREST level. It also converts the whole
    clip in one step instead of one sample at a time, so a long reply is
    ready to send sooner.
    """

    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)

    # Never 0. The board reads each piece of speech as text, which ends at
    # the first 0, so a 0 - the very bottom of the loudest sounds - used to
    # cut the rest of that piece off, up to a quarter of a second of speech.
    # One step up from the bottom cannot be heard.
    return np.clip(np.round((samples + 1.0) * 127.5), 1, 255).astype(np.uint8).tobytes()


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


# How long THEO keeps asking when the board does not answer a status
# check. One missed check is usually Wi-Fi catching its breath - in the
# 2026-10-10 session one went unanswered for three seconds and cost the
# reminder its chime. Ten seconds of silence means the board really is gone.
STATUS_GRACE = 10.0


def wait_for_speech(timeout=30, stop=None):
    """Wait until the speaker has played everything. stop() returning True ends the wait early."""
    deadline = time.monotonic() + timeout
    quiet_checks = 0
    unanswered_since = None
    while time.monotonic() < deadline:
        if stop is not None and stop():
            return
        asked_at = time.monotonic()
        try:
            response = requests.get("http://" + NODE_IP + "/audio/status", timeout=3)
            response.raise_for_status()
            status = response.json()
        except OSError:
            # requests' connection, timeout and HTTP errors are all OSErrors.
            if unanswered_since is None:
                unanswered_since = asked_at
            if time.monotonic() - unanswered_since >= STATUS_GRACE:
                raise
            time.sleep(0.25)
            continue
        unanswered_since = None
        if (type(status.get("playing")) is not bool or
                type(status.get("music_active")) is not bool or
                type(status.get("buffered_ms")) not in (int, float)):
            raise RuntimeError("The node did not return a valid speaker status")
        if status["music_active"]:
            return
        if not status["playing"] and status["buffered_ms"] == 0:
            quiet_checks += 1
            if quiet_checks == 2:
                return
        else:
            quiet_checks = 0
        time.sleep(0.05)
    raise TimeoutError("The speaker did not finish its reply in time")


# How long THEO waits to connect to the board when it sends speech, and,
# when the board could not be reached, how long it gives the board to
# answer again before one more try. The board answers at once on the same
# Wi-Fi. On 2026-10-31 it stopped answering: THEO waited 30 seconds to
# connect, took that for an ElevenLabs failure and switched to the offline
# voice until a restart, which plays through the same board.
NODE_CONNECT_TIMEOUT = 5.0
BOARD_PATIENCE = 10.0


def never_reached_board(error):
    """True when a request failed before the board could have received any of it."""

    described = f"{type(error).__name__}: {error}".casefold()

    return any(sign in described for sign in (
        "connecttimeout", "newconnectionerror", "failed to establish a new connection",
        "no route to host", "connection refused"))


def board_answers_again(patience):
    """True once the board answers its sound report, waiting at most `patience` seconds."""

    deadline = time.monotonic() + patience

    while time.monotonic() < deadline:
        if node_audio_status() is not None:
            return True
        time.sleep(1.0)

    return False


def post_piece(piece, timeout, extra):
    """
    One piece of sound to the board. A board that could not be reached gets
    BOARD_PATIENCE seconds to answer again (no longer than the sound's own
    timeout), then one more try; after that it is NodeUnavailable, so THEO
    names the board and not the voice. Any other failure is raised as it
    came: the board may have taken the piece, so it is never sent twice.
    """

    for attempt in (1, 2):
        try:
            response = requests.post(
                "http://" + NODE_IP + "/audio/speak",
                data=piece,
                headers={"Content-Type": "application/octet-stream"},
                timeout=(min(NODE_CONNECT_TIMEOUT, timeout), timeout),
                **extra
            )
            response.raise_for_status()
            return
        except Exception as error:
            if not never_reached_board(error):
                raise

            print(f"[voice] the board at {NODE_IP} did not answer: {error}")

            if attempt == 2 or not board_answers_again(min(BOARD_PATIENCE, timeout)):
                raise NodeUnavailable(f"the board at {NODE_IP} did not take the speech",
                                      "I can't reach my speaker right now.") from error

            print("[voice] the board answers again, so the speech goes once more")


def send_to_node(sound, timeout=30):
    """
    Queue small pieces, then retain the audio lock until playback finishes.

    sound is the node's 8-bit sound, or several of them one after another:
    the offline voice sends a sentence at a time, each made while the one
    before plays (see offline_sounds). It waits for the last to finish.

    The talk button stops it. The board cuts the sound off the moment the
    button goes down, and from then on turns away any piece sent before
    THEO knew about the press; this stops sending the rest. Returns False
    when the button stopped it.
    """

    global theo_quiet_since

    if isinstance(sound, (bytes, bytearray)):
        if len(sound) == 0:
            return True

        sounds = [sound]
    else:
        sounds = sound

    with AUDIO_LOCK:
        # THEO's own voice must never wake THEO: the listener ignores the
        # microphone while this is set, and for TALK_ECHO seconds after.
        THEO_TALKING.set()

        # A press from now on (or since the request THEO is answering was
        # heard) means stop.
        if talk_button is None:
            presses_before = None
        elif answering_from is not None:
            presses_before = answering_from
        else:
            presses_before = talk_button.presses()

        def pressed():
            return presses_before is not None and talk_button.presses() != presses_before

        extra = {}

        if talk_button is not None:
            extra["params"] = {"presses": talk_button.board_count()}

        # Whether the board may have some of this sound to play, and so
        # whether to wait for it to finish: a board that was never reached
        # has nothing to play. Whatever goes wrong on the board's side is
        # NodeUnavailable: the voice was made by then, so THEO names the
        # board and not the voice (Aiden's log, 2026-10-31).
        maybe_playing = False
        board_gone = False

        try:
            for part in sounds:
                for start in range(0, len(part), NODE_PIECE):
                    if pressed():
                        break
                    try:
                        post_piece(part[start:start + NODE_PIECE], timeout, extra)
                    except NodeUnavailable:
                        board_gone = True
                        raise
                    except Exception as error:
                        # The board took the piece and its answer was lost:
                        # it may be playing, so THEO still waits for it below.
                        maybe_playing = True
                        raise NodeUnavailable(f"the board at {NODE_IP} took the speech, but its "
                                              f"answer was lost: {error}",
                                              "I can't reach my speaker right now.") from error
                    maybe_playing = True

                if pressed():
                    break
        finally:
            # Stop making the rest of a reply the button cut short.
            close = getattr(sounds, "close", None)

            if close is not None:
                close()

            try:
                if maybe_playing and not board_gone:
                    try:
                        wait_for_speech(timeout, stop=pressed)
                    except (OSError, RuntimeError) as error:
                        # requests' errors and TimeoutError are OSErrors: the
                        # board stopped answering, or never finished playing.
                        raise NodeUnavailable(f"the board at {NODE_IP} stopped answering while "
                                              f"it spoke: {error}",
                                              "I can't reach my speaker right now.") from error
            finally:
                theo_quiet_since = time.monotonic()
                THEO_TALKING.clear()

        if pressed():
            stop_speaking()
            return False

    return True


def stop_speaking():
    """The talk button stopped THEO: make sure the board is silent, and remember it for this turn."""

    STOPPED_BY_BUTTON.set()

    # That press was a stop, not a request to listen.
    if talk_button is not None:
        talk_button.take()

    print("[button] talk button pressed: THEO stopped talking")

    try:
        requests.get("http://" + NODE_IP + "/audio/stop", timeout=3)
    except Exception as error:
        print(f"[button] could not silence the board: {error}")


def stopped_while_answering():
    """
    True if the talk button was pressed since the request was heard: THEO
    drops the answer before doing any more with it - no tools, no voice -
    and goes back to waiting. The press is used up, so it does not start a
    new turn afterwards.
    """

    if talk_button is None or answering_from is None:
        return False

    if talk_button.presses() == answering_from:
        return False

    talk_button.take()
    STOPPED_BY_BUTTON.set()
    print("[button] talk button pressed: THEO dropped that answer")

    return True


def node_sound(samples, sample_rate):
    """
    Any sound as exactly what the node understands. samples: numbers from -1
    to 1 (or 16-bit whole numbers), one channel or two, at any rate.
    """

    samples = np.asarray(samples)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)       # two channels -> one

    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32)

    return eight_bit_from_samples(resample_for_node(samples, sample_rate))


def play_on_node(samples, sample_rate, timeout=30):
    """Play any sound through the node's speaker - the one in the room (see node_sound)."""

    send_to_node(node_sound(samples, sample_rate), timeout=timeout)


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


# ------------------------------------------------------------------
# HOW THEO SPEAKS: SPEED AND LOUDNESS
#
# "Slower", "faster", "louder" and "quieter" change these for both voices,
# a step at a time, and they are kept in theo_settings.json next to this
# program, so THEO stays the way the person likes it after a restart.
#
# Speed is ElevenLabs' own speed setting, which goes from 0.7 to 1.2, and
# the offline voice's, which is set the same way. Loudness multiplies the
# voice; above 1, the loudest moments are eased under the speaker's limit
# instead of being clipped, so it gets louder without crackling. The
# offline voice reaches only about half the speaker's range at 1, so it
# has the most to gain.
# ------------------------------------------------------------------

SETTINGS_FILE = os.path.join(THEO_DIR, "theo_settings.json")

VOICE_SPEEDS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
VOICE_LOUDNESS = (0.7, 1.0, 1.4, 2.0)

voice_settings = {"speed": 1.0, "loudness": 1.0}


def load_voice_settings():
    """Read theo_settings.json, if it is there. Anything odd in it is ignored."""

    try:
        with open(SETTINGS_FILE) as handle:
            saved = json.load(handle)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as error:
        print(f"[voice] could not read theo_settings.json ({error}); using the usual voice")
        return

    for name, allowed in (("speed", VOICE_SPEEDS), ("loudness", VOICE_LOUDNESS)):
        value = saved.get(name) if isinstance(saved, dict) else None

        if isinstance(value, (int, float)) and min(allowed) <= value <= max(allowed):
            voice_settings[name] = min(allowed, key=lambda step: abs(step - value))

    print(f"[voice] speed {voice_settings['speed']}, loudness {voice_settings['loudness']}")


def save_voice_settings():
    try:
        part = SETTINGS_FILE + ".part"

        with open(part, "w") as handle:
            json.dump(voice_settings, handle)

        os.replace(part, SETTINGS_FILE)
    except OSError as error:
        print(f"[voice] could not save theo_settings.json: {error}")


def change_voice(name, direction):
    """
    One step slower or faster (name "speed"), quieter or louder ("loudness").
    Returns True if it changed, False if it was already at the end.
    """

    steps = VOICE_SPEEDS if name == "speed" else VOICE_LOUDNESS
    now = min(range(len(steps)), key=lambda i: abs(steps[i] - voice_settings[name]))
    wanted = now + direction

    if wanted < 0 or wanted >= len(steps):
        return False

    voice_settings[name] = steps[wanted]
    save_voice_settings()

    print(f"[voice] {name} is now {steps[wanted]}")

    return True


def louder(samples, loudness):
    """
    The sound `loudness` times as loud. Above 1, anything past 80% of full
    scale is eased smoothly toward 99% of it rather than clipped.
    """

    samples = np.asarray(samples, dtype=np.float32) * loudness

    if loudness <= 1.0:
        return samples

    knee, ceiling = 0.8, 0.99
    size = np.abs(samples)
    over = size > knee

    samples[over] = np.sign(samples[over]) * (
        knee + (ceiling - knee) * np.tanh((size[over] - knee) / (ceiling - knee)))

    return samples


def speak(reply, offline=False):
    """THEO's voice: ElevenLabs, or the offline voice (Kokoro) when offline=True."""

    # No formatting stars, emoji or picture symbols in the voice.
    reply = speakable(reply)

    if not reply:
        return

    speed = voice_settings["speed"]
    loudness = voice_settings["loudness"]

    if not offline:
        request = dict(
            text=reply,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="pcm_16000"
        )

        try:
            # Only when it differs, so the usual voice is asked for exactly as before.
            if speed != 1.0:
                request["voice_settings"] = (VoiceSettings(speed=speed) if VoiceSettings is not None
                                             else {"speed": speed})

            audio = el_client.text_to_speech.convert(**request)
        except (TypeError, ValueError) as error:
            if speed == 1.0:
                raise
            # An ElevenLabs library too old for the speed setting refuses it
            # before anything is sent. Speak at the usual speed instead.
            print(f"[voice] this ElevenLabs library cannot change speed ({error}); "
                  "updating it (pip install -U elevenlabs) makes slower work here too")
            request.pop("voice_settings", None)
            audio = el_client.text_to_speech.convert(**request)

        pcm = b"".join(audio)

        sound = to_eight_bit(pcm, loudness)

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
        # THE OFFLINE VOICE, a sentence at a time, played through the node
        # like everything else (see offline_sounds).
        send_to_node(offline_sounds(reply, speed, loudness))


# A sentence shorter than this is spoken together with the next one, so the
# offline voice does not stop and start every few words.
SENTENCE_SHORTEST = 30


def sentences_of(text):
    """The reply in sentences, for the offline voice: "Hello! How can I help?" stays together."""

    sentences = []

    # A sentence can end inside quotes or a bracket: in Aiden's log of
    # 2026-10-31, 'tap “Save.” If no edit option appears' was not split, so
    # the offline voice made a whole 22-second reply before any of it played.
    for part in re.split(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+(?=[\"'“‘(]?[A-Z0-9])",
                         text.strip()):
        part = part.strip()

        if not part:
            continue

        if sentences and len(sentences[-1]) < SENTENCE_SHORTEST:
            sentences[-1] += " " + part
        else:
            sentences.append(part)

    return sentences


# A long first sentence of an offline reply is spoken in two parts: up to
# its first comma, semicolon, colon or dash, then the rest. On the Uno Q
# Kokoro needs about 3.5 seconds of work for each second of speech (Aiden's
# log, 2026-10-31), so a first sentence 8.5 seconds long kept him waiting
# 30 seconds for THEO's first word.
FIRST_PHRASE_LONGEST = 60
FIRST_PHRASE_SHORTEST = 12


def first_phrase(sentence):
    """A long sentence as (its first phrase, the rest), or (sentence, None) when it stays whole."""

    if len(sentence) <= FIRST_PHRASE_LONGEST:
        return sentence, None

    for match in re.finditer(r"(?<=[,;:])\s+|\s+[-–—]\s+", sentence):
        head = sentence[:match.start()].strip()
        rest = sentence[match.end():].strip()

        if len(head) > FIRST_PHRASE_LONGEST:
            break

        if len(head) >= FIRST_PHRASE_SHORTEST and rest:
            return head, rest

    return sentence, None


def offline_sounds(reply, speed, loudness):
    """
    The offline voice saying the reply, as the node's sound for send_to_node:
    the first sentence, made now, then each one after it, made while the one
    before plays. On the Uno Q a whole reply used to take up to a minute to
    make before any of it was heard (until 2026-10-29.1); now the first
    sentence plays as soon as it is ready, and a long first sentence starts
    with its first phrase (see first_phrase). How long each sentence took to
    make is printed beside how long it lasts.
    """

    sentences = sentences_of(reply)

    if sentences:
        head, rest = first_phrase(sentences[0])

        if rest:
            sentences[0:1] = [head, rest]

    def make(sentence):
        started = time.monotonic()

        samples, sample_rate = offline_voice.create(
            sentence,
            voice=KOKORO_VOICE,
            speed=speed,
            lang="en-us"
        )

        print(f"[voice] offline voice: {len(samples) / sample_rate:.1f} s of speech, "
              f"made in {time.monotonic() - started:.1f} s")

        if loudness != 1.0:
            samples = louder(samples, loudness)

        return node_sound(samples, sample_rate)

    if not sentences:
        return iter(())

    first = make(sentences[0])
    made = queue.Queue()
    stop = threading.Event()

    def make_the_rest():
        try:
            for sentence in sentences[1:]:
                if stop.is_set():
                    break
                made.put((make(sentence), None))
        except Exception as error:
            made.put((None, error))
        finally:
            made.put((None, None))

    if len(sentences) > 1:
        threading.Thread(target=make_the_rest, name="theo-offline-voice", daemon=True).start()

    def pieces():
        try:
            yield first

            if len(sentences) == 1:
                return

            while True:
                sound, problem = made.get()

                if problem is not None:
                    raise problem

                if sound is None:
                    return

                yield sound
        finally:
            stop.set()

    return pieces()


# After ElevenLabs turns a request down over credits or a quota, THEO waits
# this long before asking it again. It used to wait 15 minutes, so once the
# problem was fixed THEO could stay quiet for a quarter of an hour. A refused
# request costs no credits, so asking again after a minute is cheap.
VOICE_RETRY_AFTER = 60.0

# After ElevenLabs fails this many times in a row, THEO speaks with the
# offline voice until THEO restarts, instead of trying ElevenLabs, waiting,
# and switching voices partway through a conversation (Aiden, 2026-10-14).
# A reply ElevenLabs does speak starts the count again. This only happens
# when the offline voice is switched on; without it THEO keeps asking
# ElevenLabs, as before.
ELEVENLABS_FAILURES_BEFORE_OFFLINE = 2


def say(text, force=False):
    """
    Say something, and never let the voice layer kill the conversation.

    Always prints. Speaks only when speaking is worth paying for.

    Pass force=True to speak something even if it was just said.

    The audio lock covers generation, every audio chunk, and completion.
    Timer threads cannot enter halfway through another sentence.

    If ElevenLabs turns a request down over credits or a quota, THEO prints
    exactly what ElevenLabs said and does not ask it again for
    VOICE_RETRY_AFTER seconds. Meanwhile it speaks with the offline voice if
    that is switched on, and otherwise only prints. With the offline voice
    on, any other ElevenLabs failure - no internet, say - is handled the same
    way, and with no internet at start-up the offline voice speaks first.

    When ElevenLabs has failed ELEVENLABS_FAILURES_BEFORE_OFFLINE times in a
    row, the offline voice becomes THEO's voice, and ElevenLabs is not asked
    again until THEO restarts.
    """

    # How long the same exact sentence must wait before it may be spoken
    # again. 300 seconds = 5 minutes. Raise it to be stricter.
    REPEAT_COOLDOWN = 300

    with AUDIO_LOCK:
        now = time.monotonic()

        # Something new to say: an earlier press of the talk button no
        # longer applies.
        STOPPED_BY_BUTTON.clear()

        if not hasattr(say, "spoken_at"):
            say.spoken_at = {}
            say.blocked_until = 0.0 if ONLINE else now + VOICE_RETRY_AFTER
            say.failures_in_a_row = 0
            say.offline_only = False

        print(f"THEO: {text}")

        if say.offline_only or now < say.blocked_until:
            speak_offline(text)
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
        except NodeUnavailable as error:
            # ElevenLabs made the voice and the board did not take it (see
            # send_to_node), so ElevenLabs did not fail: its count starts again,
            # and the offline voice would not help, as it plays through the
            # same board. The reply is printed above, and may be said again.
            say.spoken_at.pop(text, None)
            say.failures_in_a_row = 0
            print(f"[voice] {error}, so THEO could only print that reply")
        except Exception as error:
            detail = str(error)
            lowered = detail.lower()
            say.failures_in_a_row += 1

            if (offline_voice is not None and
                    say.failures_in_a_row >= ELEVENLABS_FAILURES_BEFORE_OFFLINE):
                if "quota" in lowered or "credits remaining" in lowered:
                    print(f"[voice] ElevenLabs turned this down: {detail}")
                else:
                    print(f"[voice unavailable: {detail}]")

                say.offline_only = True
                print(f"[voice] ElevenLabs failed {say.failures_in_a_row} times in a row, so THEO "
                      "speaks with the offline voice from now on. Restart THEO to use "
                      "ElevenLabs again.")
                speak_offline(text)
            elif "quota" in lowered or "credits remaining" in lowered:
                say.blocked_until = now + VOICE_RETRY_AFTER
                print(f"[voice] ElevenLabs turned this down: {detail}")
                print("[voice] (A credit limit on the API key itself gives this too, even "
                      "when the account still has credits.) Asking ElevenLabs again in "
                      f"{VOICE_RETRY_AFTER:.0f} seconds.")
                speak_offline(text)
            elif offline_voice is not None:
                say.blocked_until = now + VOICE_RETRY_AFTER
                print(f"[voice unavailable: {detail}]")
                print("[voice] Using the offline voice. Asking ElevenLabs again in "
                      f"{VOICE_RETRY_AFTER:.0f} seconds.")
                speak_offline(text)
            else:
                print(f"[voice unavailable: {detail}]")
        else:
            say.failures_in_a_row = 0


def speak_offline(text):
    """While ElevenLabs is unavailable: the offline voice if it is switched on."""

    if offline_voice is None:
        return

    try:
        speak(text, offline=True)
    except NodeUnavailable as error:
        print(f"[voice] {error}, so THEO could only print that reply")
    except Exception as error:
        print(f"[voice unavailable: {error}]")


# ------------------------------------------------------------------
# THE OFFLINE BRAIN (Ollama)
#
# When Groq cannot be reached, THEO answers with Ollama instead, running on
# this Uno Q or on a computer on the same Wi-Fi.
#
# App Lab runs this program in a container of its own, so "localhost" here
# is that container, not the Uno Q. An Ollama on the Uno Q is reached
# through the container's gateway instead, and only answers there if it
# listens beyond the Uno Q itself (OLLAMA_HOST=0.0.0.0 in Ollama's service).
#
# Settings, both optional, in key.env:
#     OLLAMA_URL=http://192.168.0.20:11434   Ollama on another computer
#     OLLAMA_MODEL=llama3.2:3b               the model (llama3.2:1b otherwise)
#
# llama3.2:1b is the default from 2026-10-19.1: it is the one that suits the
# 4 GB Uno Q, and the one Aiden pulled. Until then the default was 3b, so
# without OLLAMA_MODEL in key.env THEO found Ollama but could not use it.
#
# Without OLLAMA_MODEL, THEO uses the first of OLLAMA_PREFERRED that Ollama
# has. The q4_K_M build of llama3.2:1b is the same model stored more
# compactly: loaded by Ollama 0.40.13 with OLLAMA_CONTEXT, it takes about
# 1.2 GB of memory, against 1.7 GB for llama3.2:1b. Get it with
#     ollama pull llama3.2:1b-instruct-q4_K_M
#
# Ollama is looked for once, at start-up, so start it before THEO.
# ------------------------------------------------------------------

OLLAMA_PORT = 11434
OLLAMA_DEFAULT_MODEL = "llama3.2:1b"
OLLAMA_PREFERRED = ("llama3.2:1b-instruct-q4_K_M", OLLAMA_DEFAULT_MODEL)
ROUTES_FILE = "/proc/net/route"

# A small computer can take a long while to answer. It is given that time,
# but not asked twice.
OLLAMA_TIMEOUT = 300.0


# ------------------------------------------------------------------
# WHAT THE OFFLINE MODEL IS SENT (2026-10-29.1)
#
# Before it writes a word, Ollama reads what it is sent. Until 2026-10-29.1
# every question sent it THEO's full instructions and all sixteen tools,
# which go in with the question itself, and the notes about the person,
# which change from question to question, near the start: Ollama read
# about 1,700 to 2,100 tokens for every question (measured with Ollama
# 0.40.13 and llama3.2:1b; the new way, 94 to 278). Now it is sent:
#
#   - short instructions of its own, the same every time;
#   - only the tools the question's words call for (OFFLINE_TOOLS_BY_WORD);
#   - the notes, and any other word to the model, together with the
#     question, at the end, so the start stays the same;
#   - the last few exchanges, from a starting point that only moves now
#     and then (see offline_window).
#
# It talks to Ollama's own /api/chat rather than its OpenAI-style address,
# so that it can keep the model loaded for OLLAMA_KEEP_ALIVE after each
# answer, give it room for OLLAMA_CONTEXT tokens (less memory than
# Ollama's own setting) and cap an answer at OLLAMA_LONGEST_ANSWER tokens.
# Each answer prints how much Ollama read and wrote, and how long each took.
# ------------------------------------------------------------------

OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_CONTEXT = 2048
OLLAMA_LONGEST_ANSWER = 160
OFFLINE_EXCHANGES_LEAST = 2
OFFLINE_EXCHANGES_MOST = 5

OFFLINE_INSTRUCTIONS = (
    "You are THEO, a kind home helper for older people who are new to technology. The user lives "
    "in Kingston, Jamaica. Everything you write is spoken aloud, so answer in one or two short, "
    "plain sentences with no lists or symbols, and give instructions one step at a time. Use a "
    "tool only for what the user asks, and never say you set a reminder, timer or alarm, or that "
    "you will remember something, unless you use its tool.")

# The tools a question can use offline, by the words in it. A question with
# none of these words is answered in words alone, which on the Uno Q is the
# quickest answer of all.
OFFLINE_TOOLS_BY_WORD = (
    (r"\b(?:time|clock|date|day|today|o'?clock)\b", ("get_time",)),
    (r"\b(?:timers?|alarms?|remind|reminders?|minutes?|hours?|seconds?|wake me|tablets?|pills?|"
     r"medicines?|medication|every (?:morning|night|day|evening))\b",
     ("set_timer", "set_reminder", "set_reminder_at", "list_reminders", "cancel_reminder")),
    (r"\b(?:remember|forget|forgot|recall|note)\b|\bcall me\b|"
     r"\bmy (?:name|son|daughter|grand\w*|wife|husband|sister|brother|phone|tablet|computer|"
     r"laptop|tv|doctor)(?:'s| is| are| was)\b",
     ("remember", "recall", "forget")),
    (r"\b(?:weather|rain|raining|sunny|hot|cold|temperature|forecast|umbrella)\b", ("get_weather",)),
    (r"\b(?:play|music|songs?|tracks?|spotify|pause|resume|skip|next|previous)\b",
     ("play_spotify", "pause_spotify", "next_track", "previous_track")),
    (r"\b(?:search|look up|google|news|who is|who was)\b", ("web_search",)),
    (r"\b(?:error|screen|camera|picture|photo|look at|see this|read this)\b", ("analyze_error",)),
)

# Where the offline window starts (see offline_window).
offline_window_start = {"first": None}


def tool_names_for(words):
    """The names of the tools these words call for (see OFFLINE_TOOLS_BY_WORD)."""

    words = (words or "").casefold().replace("’", "'")
    names = set()

    for pattern, group in OFFLINE_TOOLS_BY_WORD:
        if re.search(pattern, words):
            names.update(group)

    return names


def offline_window(exchanges):
    """
    The exchanges Ollama is sent. They run from a starting point that stays
    put while the window grows to OFFLINE_EXCHANGES_MOST exchanges, then
    moves on to the last OFFLINE_EXCHANGES_LEAST. Ollama does not read again
    what is the same at the start of a request as the last time, so most
    questions only cost their own words.
    """

    start = offline_window_start["first"]

    for index, exchange in enumerate(exchanges):
        if exchange and exchange[0] is start:
            if len(exchanges) - index <= OFFLINE_EXCHANGES_MOST:
                return exchanges[index:]
            break

    kept = exchanges[-OFFLINE_EXCHANGES_LEAST:]
    offline_window_start["first"] = kept[0][0] if kept and kept[0] else None

    return kept


def native_message(message, tool_names):
    """One message in the form Ollama's /api/chat takes."""

    role = message.get("role")
    content = message.get("content")
    native = {"role": role, "content": content if isinstance(content, str) else ""}

    if role == "assistant" and message.get("tool_calls"):
        calls = []

        for call in message["tool_calls"]:
            function = call.get("function") or {}
            arguments = function.get("arguments")

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except ValueError:
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            tool_names[call.get("id")] = function.get("name")
            calls.append({"function": {"name": function.get("name") or "", "arguments": arguments}})

        native["tool_calls"] = calls

    if role == "tool" and tool_names.get(message.get("tool_call_id")):
        native["tool_name"] = tool_names[message.get("tool_call_id")]

    return native


def offline_request(messages, tools):
    """What Ollama is sent for one request (see WHAT THE OFFLINE MODEL IS SENT): (messages, tools)."""

    extras = []
    talk = []

    for message in messages[1:]:
        if isinstance(message, dict) and message.get("role") == "system":
            extras.append(message.get("content") or "")
        elif isinstance(message, dict):
            talk.append(message)

    exchanges = []

    for message in talk:
        if message.get("role") == "user" or not exchanges:
            exchanges.append([])
        exchanges[-1].append(message)

    tool_names = {}
    sent = [{"role": "system", "content": OFFLINE_INSTRUCTIONS}]
    sent += [native_message(message, tool_names)
             for exchange in offline_window(exchanges) for message in exchange]

    asked = next((message for message in reversed(sent) if message["role"] == "user"), None)
    words = asked["content"] if asked else ""

    # Words to the model other than the notes - "you used no tool for that" -
    # can call for tools too.
    for extra in extras:
        if not extra.startswith(NOTES_HEADING):
            words += " " + extra

    if asked is not None and extras:
        asked["content"] = "\n\n".join(extras) + "\n\nWhat they said: " + asked["content"]

    wanted = tool_names_for(words)
    chosen = [tool for tool in tools or ()
              if isinstance(tool, dict) and tool.get("function", {}).get("name") in wanted]

    return sent, chosen


def tool_call_written_out(text, offered):
    """
    A small model sometimes writes its tool call out as text instead of
    making it: {"name": "set_timer", "parameters": {...}}. For a tool it was
    offered, that is taken as the call rather than spoken aloud.
    """

    written = (text or "").strip().strip("`").strip()

    if written.startswith("json"):
        written = written[4:].strip()

    if not (written.startswith("{") and written.endswith("}")):
        return None

    try:
        data = json.loads(written)
    except ValueError:
        return None

    names = {tool["function"]["name"] for tool in offered or () if isinstance(tool, dict)}

    if not isinstance(data, dict) or data.get("name") not in names:
        return None

    arguments = data.get("parameters", data.get("arguments", {}))

    return SimpleNamespace(id="offline_1", type="function", function=SimpleNamespace(
        name=data["name"], arguments=json.dumps(arguments if isinstance(arguments, dict) else {})))


def whole_sentences(text):
    """An answer Ollama had to stop partway through, cut back to its last whole sentence."""

    ends = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", text or "")]

    return text[:ends[-1]].strip() if ends else (text or "").strip()


def ollama_timing(data, offered):
    """One line: how much Ollama read and wrote for this answer, and how long each took."""

    def seconds(key):
        return (data.get(key) or 0) / 1e9

    sent, wrote = data.get("prompt_eval_count") or 0, data.get("eval_count") or 0
    reused = data.get("prompt_eval_cached_count")

    # Ollama 0.40 counts the whole request in prompt_eval_count, and how much
    # of it was reused from the answer before in prompt_eval_cached_count:
    # only the rest is read again. Without that field, the count is printed
    # as Ollama gives it.
    if reused is None:
        line = f"[offline] Ollama read {sent} tokens in {seconds('prompt_eval_duration'):.1f} s"
    else:
        line = (f"[offline] Ollama read {sent - reused} new tokens in "
                f"{seconds('prompt_eval_duration'):.1f} s ({reused} reused)")

    line += f" and wrote {wrote} in {seconds('eval_duration'):.1f} s"

    if wrote and seconds("eval_duration") > 0:
        line += f" ({wrote / seconds('eval_duration'):.1f} a second)"

    if seconds("load_duration") >= 1.0:
        line += f"; loading the model took {seconds('load_duration'):.1f} s"

    return line + f"; {offered} tool(s) offered"


def ollama_answer(data, offered):
    """Ollama's answer, shaped like Groq's: answer.choices[0].message.content and .tool_calls."""

    message = data.get("message") or {}
    text = message.get("content") or ""
    calls = []

    for number, call in enumerate(message.get("tool_calls") or [], start=1):
        function = call.get("function") or {}
        arguments = function.get("arguments")

        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})

        calls.append(SimpleNamespace(id=call.get("id") or f"offline_{number}", type="function",
                                     function=SimpleNamespace(name=function.get("name") or "",
                                                              arguments=arguments)))

    if not calls:
        written = tool_call_written_out(text, offered)

        if written is not None:
            calls, text = [written], ""

    if not calls and data.get("done_reason") == "length":
        text = whole_sentences(text)

    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=text, tool_calls=calls or None))])


def ollama_chat(address, model, messages, tools=None, tool_choice=None, temperature=0.7,
                max_tokens=None, **unused):
    """One answer from Ollama, sent the offline way (see WHAT THE OFFLINE MODEL IS SENT)."""

    sent, chosen = offline_request(messages, None if tool_choice == "none" else tools)
    longest = OLLAMA_LONGEST_ANSWER if max_tokens is None else min(max_tokens, OLLAMA_LONGEST_ANSWER)
    body = {"model": model, "messages": sent, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": OLLAMA_CONTEXT, "num_predict": longest,
                        "temperature": temperature}}

    if chosen:
        body["tools"] = chosen

    reply = requests.post(address + "/api/chat", json=body, timeout=OLLAMA_TIMEOUT)

    if reply.status_code >= 400:
        raise RuntimeError(f"Ollama answered {reply.status_code}: {reply.text}")

    data = reply.json()
    print(ollama_timing(data, len(chosen)))

    return ollama_answer(data, chosen)


def ollama_client(address):
    """Ollama at `address`, asked the way Groq is: client.chat.completions.create(...)."""

    def create(**request):
        return ollama_chat(address, **request)

    return SimpleNamespace(address=address,
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def warm_up_ollama():
    """
    Starting without the internet: Ollama loads its model now and reads
    THEO's offline instructions, so the first question is not kept waiting.
    """

    local_client, local_model = ollama_brain

    try:
        local_client.chat.completions.create(
            model=local_model, messages=[{"role": "system", "content": ""},
                                         {"role": "user", "content": "Hello."}],
            temperature=0.0, max_tokens=1)
        print(f"[offline] Ollama has {local_model} loaded and ready.")
    except Exception as error:
        print(f"[offline] Ollama could not load {local_model} ahead of time: {error}")


def container_gateway():
    """The address of the machine this container runs on (the Uno Q), or None."""

    try:
        with open(ROUTES_FILE) as routes:
            for line in list(routes)[1:]:
                fields = line.split()
                if len(fields) > 2 and fields[1] == "00000000" and fields[2] != "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, ValueError):
        pass

    return None


def ollama_addresses():
    """Where to look for Ollama: OLLAMA_URL if it is set, otherwise here and the Uno Q."""

    configured = (os.environ.get("OLLAMA_URL") or "").strip().rstrip("/")

    if configured:
        return [configured]

    addresses = [f"http://localhost:{OLLAMA_PORT}"]
    gateway = container_gateway()

    if gateway:
        addresses.append(f"http://{gateway}:{OLLAMA_PORT}")

    return addresses


def find_ollama():
    """Ollama and the model it answers with, as (client, model), or None if it is not ready."""

    configured = (os.environ.get("OLLAMA_MODEL") or "").strip()
    addresses = ollama_addresses()

    for address in addresses:
        try:
            reply = requests.get(address + "/api/tags", timeout=2)
            reply.raise_for_status()
            names = [entry["name"] for entry in reply.json()["models"]]
        except Exception:
            continue

        candidates = (configured,) if configured else OLLAMA_PREFERRED
        wanted = next((model for model in candidates
                       if model in names or model + ":latest" in names), None)

        if wanted is None:
            missing = configured or OLLAMA_DEFAULT_MODEL
            pull = configured or OLLAMA_PREFERRED[0]
            print(f"[offline] Ollama answers at {address} but does not have {missing} "
                  f"(it has: {', '.join(names) or 'no models'}). Run 'ollama pull "
                  f"{pull}' there, or set OLLAMA_MODEL in key.env.")
            return None

        print(f"[offline] Ollama: ready at {address} with {wanted}. THEO answers "
              "with it whenever Groq can't be reached.")

        # Only while THEO chooses the model and the compact build is missing.
        # Aiden's log of 2026-10-31 printed this with the compact build in
        # use, after OLLAMA_DEFAULT_MODEL had been set to its name.
        if (not configured and OLLAMA_PREFERRED[0] not in names and
                OLLAMA_PREFERRED[0] + ":latest" not in names):
            print(f"[offline] Tip: 'ollama pull {OLLAMA_PREFERRED[0]}' gets the same model in "
                  "less memory (about 1.2 GB instead of 1.7 GB); THEO uses it by itself once "
                  "it is there.")

        return ollama_client(address), wanted

    print("[offline] Ollama: not found at " + " or ".join(addresses) + ", so without "
          "the internet THEO can't answer. If Ollama runs on this Uno Q, it has to "
          "listen beyond the Uno Q itself (OLLAMA_HOST=0.0.0.0). If it runs on a "
          "computer, put OLLAMA_URL=http://<its address>:11434 in key.env.")
    return None


def think(**request):
    """
    The language model's answer to one request: Groq while it works, and
    Ollama when Groq cannot be reached or turns the request down. Without
    Ollama, Groq's problem is passed on as it always was.
    """

    global groq_down_until

    problem = None

    if ollama_brain is None or time.monotonic() >= groq_down_until:
        groq = client if ollama_brain is None else client.with_options(
            max_retries=0, timeout=GROQ_TIMEOUT)

        try:
            answer = groq.chat.completions.create(model=ONLINE_MODEL, **request)
        except Exception as error:
            if ollama_brain is None:
                if describe_problem(error)[1]:
                    brain_lost()
                raise
            problem = error
        else:
            brain_reached()
            return answer

    if problem is not None:
        print(f"[offline] Groq could not answer ({problem}). Asking Ollama instead.")

        if describe_problem(problem)[1]:
            groq_down_until = time.monotonic() + GROQ_RETRY_AFTER
            print(f"[offline] Asking Groq again in {GROQ_RETRY_AFTER:.0f} seconds.")
            brain_lost()

    local_client, local_model = ollama_brain
    print(f"[offline] answering with Ollama ({local_model})")
    started = time.monotonic()
    answer = local_client.chat.completions.create(model=local_model, **request)
    print(f"[offline] Ollama answered in {time.monotonic() - started:.1f} s")
    return answer


def plain_message(message, tool_calls):
    """
    The model's tool requests as plain data, so that whichever model answers
    next - Groq or Ollama - can read them back. THEO carries out every one
    of them, so every tool request in the conversation has its result.
    """

    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {"name": tool_call.function.name,
                         "arguments": tool_call.function.arguments},
        } for tool_call in tool_calls],
    }


# ------------------------------------------------------------------
# THEO'S CLOCK
#
# Jamaica time, on purpose, whatever time zone the Uno Q or App Lab's
# container is set to, so "8 in the morning" is 8 in the morning in
# Kingston. Jamaica keeps UTC-5 all year; if this Uno Q has no time zone
# data, that is used directly. THEO prints its time when it starts.
# ------------------------------------------------------------------

TIME_ZONE = "America/Jamaica"
TIME_ZONE_HOURS = -5


def theo_timezone():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(TIME_ZONE)
    except Exception:
        return timezone(timedelta(hours=TIME_ZONE_HOURS), TIME_ZONE)


THEO_TZ = theo_timezone()


def local_now():
    """Now, in Jamaica."""
    return datetime.now(THEO_TZ)


def get_time():
    return local_now().strftime("%I:%M %p") #formats time it terms of hour,minute, AM/PM


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


# SPOTIFY PLAYS THROUGH THEO'S OWN SPEAKER.
#
# theo_spotify_bridge.py runs on the Uno Q, outside App Lab, and shows up in
# Spotify as a speaker called "THEO". Every song THEO plays goes there
# instead of to your laptop.
#
# While a song plays, THEO keeps listening but only answers when you say its
# name - "Theo, pause the music" - so the words in the song cannot set it
# off. When THEO answers, the song pauses and then carries on, unless what
# you asked for was to change the music.

THEO_SPEAKER_NAME = "THEO"      # must match SPEAKER_NAME in theo_spotify_bridge.py

THEO_SPEAKER_MISSING = ("I can't find my speaker in Spotify right now. It may "
                        "still be starting up, so please ask me again in a minute.")


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
    if "device not found" in text:
        return THEO_SPEAKER_MISSING
    return "I couldn't play that through my speaker. Please try again in a moment."


def is_theo_speaker(device):
    return (device.get("name") or "").strip().lower() == THEO_SPEAKER_NAME.lower()


def get_playback_device():
    """THEO's own speaker, as Spotify sees it. Returns its device id, or None."""
    for device in sp.devices().get("devices", []):
        if is_theo_speaker(device) and not device.get("is_restricted"):
            return device["id"]

    print(f'[spotify] no speaker called "{THEO_SPEAKER_NAME}" in Spotify. Is '
          f"theo_spotify_bridge running on the Uno Q? (journalctl -u theo-spotify)")
    return None


def node_audio_status():
    """The node's sound report, or None if it does not answer quickly."""
    try:
        reply = requests.get("http://" + NODE_IP + "/audio/status", timeout=2)
        reply.raise_for_status()
        return reply.json()
    except Exception:
        return None


def music_on_speaker(status):
    """Require the name until the node and player both confirm no music."""
    if status is None or type(status.get("music_active")) is not bool:
        return True
    if status["music_active"] or time.monotonic() < music_turn.expected_until:
        return True
    if not spotify_linked():
        return False
    try:
        playback = sp.current_playback()
        return bool(playback and playback.get("is_playing") and
                    is_theo_speaker(playback.get("device") or {}))
    except Exception as error:
        print(f"[music] cannot confirm player state; say Theo first: {error}")
        return True


def wait_for_music_to_stop(limit=3.0):
    """After pausing a song: wait until the speaker has actually gone quiet."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        status = node_audio_status()
        if status is not None and status.get("music_active") is False:
            return True
        time.sleep(0.2)
    return False


# Whisper writes THEO's name a few different ways.
THEO_NAME = r"(?:theo|teo|tio|thio|theyo|thea|theodore)"

NAME_FIRST = re.compile(
    r"^\W*(?:(?:hey|hi|hello|okay|ok|so|excuse me)\W+)?" + THEO_NAME + r"\b\W*",
    re.IGNORECASE)

NAME_LAST = re.compile(r"\W+" + THEO_NAME + r"\W*$", re.IGNORECASE)


def after_name(text):
    """
    What was asked, once THEO's name is taken off.

        "Theo, pause the music"   ->  "pause the music"
        "Pause the music, Theo"   ->  "Pause the music"
        "Hey Theo."               ->  ""       (just the name)
        anything else             ->  None     (not spoken to THEO)
    """
    match = NAME_FIRST.match(text)
    if match:
        return text[match.end():].strip()

    match = NAME_LAST.search(text)
    if match:
        return text[:match.start()].strip()

    return None


def is_exit_request(text):
    addressed = after_name(text)
    if addressed is not None:
        text = addressed
    words = " ".join(re.findall(r"[a-z]+", text.casefold()))
    # The word may come twice: on 2026-10-19 "Quit. Quit." went to the
    # language model instead of stopping THEO.
    return bool(re.fullmatch(
        r"(?:(?:okay|ok|alright|all right|please) )*"
        r"(?:quit|exit|goodbye|shut down)(?: (?:quit|exit|goodbye|shut down))*"
        r"(?: (?:please|now|thank you))*", words))


# ------------------------------------------------------------------
# "SAY THAT AGAIN", "SLOWER", "LOUDER"
#
# Caught here, the way "quit" is, so they work at once and without the
# internet, and never go to the language model. "Say that again" repeats
# THEO's last answer or reminder. "Slower" and "louder" change THEO's
# voice from now on (see HOW THEO SPEAKS) and say the last answer again
# that way; "faster" and "quieter" go back the other way. While a song is
# playing, "louder" and "quieter" are left to the language model, since
# they are probably about the music.
# ------------------------------------------------------------------

COMMAND_START = (r"(?:(?:okay|ok|alright|all right|please|so|um|uh|oh|hey|sorry|excuse me|"
                 r"theo|can you|could you|would you|will you)\s)*")
COMMAND_END = r"(?:\s(?:please|theo|thanks|thank you|now|again|for me))*"

VOICE_COMMANDS = (
    ("again", r"say (?:that|it|this|what you said) again|say again|"
              r"repeat (?:that|it|this|yourself|what you said)|repeat|again|"
              r"what did you (?:just )?say|what was that|what|pardon(?: me)?|sorry|"
              r"come again|excuse me|i didnt (?:hear|catch|get) (?:that|you|it|what you said)|"
              r"one more time"),
    ("slower", r"(?:(?:speak|talk|say (?:that|it)(?: again)?|go)\s)?(?:a (?:bit|little|lot)\s)?"
               r"(?:slower|more slowly|slowly)|slow down|"
               r"(?:(?:youre|you are)\s)?(?:talking|speaking|going)\stoo fast|too fast"),
    ("faster", r"(?:(?:speak|talk|say (?:that|it)(?: again)?|go)\s)?(?:a (?:bit|little|lot)\s)?"
               r"(?:faster|quicker|more quickly)|speed up|"
               r"(?:(?:youre|you are)\s)?(?:talking|speaking|going)\stoo slow(?:ly)?|too slow"),
    ("louder", r"(?:(?:speak|talk|say (?:that|it)(?: again)?)\s)?(?:a (?:bit|little|lot)\s)?louder|"
               r"speak up|i cant hear you|i cannot hear you|(?:(?:youre|you are)\s)?too (?:quiet|soft)"),
    ("quieter", r"(?:(?:speak|talk|say (?:that|it)(?: again)?)\s)?(?:a (?:bit|little|lot)\s)?"
                r"(?:quieter|softer|more quietly|more softly)|not so loud|"
                r"(?:(?:youre|you are)\s)?too loud"),
)

VOICE_CHANGES = {
    "slower": ("speed", -1), "faster": ("speed", 1),
    "louder": ("loudness", 1), "quieter": ("loudness", -1),
}

VOICE_LIMITS = {
    "slower": "That's as slowly as I can speak.",
    "faster": "That's as fast as I speak.",
    "louder": "That's as loud as my voice goes.",
    "quieter": "That's as quietly as I speak.",
}

VOICE_CONFIRMS = {
    "slower": "Okay, I'll speak more slowly.",
    "faster": "Okay, I'll speak a little faster.",
    "louder": "Okay, I'll speak louder.",
    "quieter": "Okay, I'll speak more quietly.",
}


def voice_command(text):
    """'again', 'slower', 'faster', 'louder' or 'quieter' - or None for anything else."""

    addressed = after_name(text)

    if addressed is not None:
        text = addressed

    words = " ".join(re.findall(r"[a-z]+", text.casefold().replace("'", "").replace("’", "")))

    for name, pattern in VOICE_COMMANDS:
        if re.fullmatch(COMMAND_START + r"(?:" + pattern + r")" + COMMAND_END, words):
            return name

    return None


def answer_here(text):
    """
    What THEO answers itself, without the language model: "say that
    again", "slower", "louder" and the rest, "eyes on" and "eyes off", and
    the yes or no to a medicine reminder's question. None for anything else.
    """

    wanted = eyes_command(text)

    if wanted is not None:
        reply = turn_eyes_on() if wanted == "on" else turn_eyes_off()
        LAST_SAID["text"] = reply
        return reply

    command = voice_command(text)

    if command in ("louder", "quieter") and music_turn.paused_device is not None:
        command = None

    if command == "again":
        return LAST_SAID["text"] or "I haven't said anything yet."

    if command is not None:
        name, direction = VOICE_CHANGES[command]

        if not change_voice(name, direction):
            return VOICE_LIMITS[command]

        return LAST_SAID["text"] or VOICE_CONFIRMS[command]

    reply = answer_medicine(text)

    if reply is not None:
        LAST_SAID["text"] = reply

    return reply


def shutdown_theo():
    STOPPING.set()
    eyes_off()
    with AUDIO_LOCK:
        with REMINDER_LOCK:
            for timer in list(active_timers.values()) + list(pending_alarms):
                timer.cancel()
            active_timers.clear()
            pending_alarms.clear()
            # The reminders at a time of day stay in reminders.json for next
            # time; their clock only has to stop waiting.
            REMINDER_CLOCK.notify_all()
        music_turn.plan = None
        music_turn.paused_device = None
        music_turn.expected_until = 0.0
        music_turn.waiting_for_answer = False
        try:
            if spotify_linked():
                device = get_playback_device()
                if device:
                    sp.pause_playback(device_id=device)
        except Exception as error:
            print(f"[quit] could not pause Spotify: {error}")
        try:
            response = requests.get("http://" + NODE_IP + "/audio/stop", timeout=3)
            response.raise_for_status()
        except Exception as error:
            print(f"[quit] could not silence the node: {error}")
        try:
            # Directly, not through the ring's helper thread: the program
            # is about to end and that thread with it.
            requests.get("http://" + NODE_IP + "/ring?state=off", timeout=2)
        except Exception as error:
            print(f"[quit] could not switch the ring off: {error}")
    print("THEO: Goodbye! Restart the app when you want to talk again.")


class MusicTurn:
    """
    What happens to the song while THEO deals with one request.

    begin() pauses a song playing on THEO's speaker as soon as THEO knows
    it is being spoken to. The Spotify tools leave a plan. finish(), once
    THEO has spoken, carries the plan out - or, with no plan, simply
    carries on with the song.

    Changes are saved until THEO has spoken, so a new song does not start
    underneath THEO's voice and lose its first few seconds.

    When THEO's answer is itself a question ("Any particular song you'd
    like?"), wait_for_answer() keeps the paused song paused, so the answer
    is heard in a quiet room and does not need THEO's name. The song then
    carries on, or changes, once that answer has been dealt with, or once
    nobody answers.
    """

    def __init__(self):
        self.paused_device = None    # THEO's speaker, if begin() paused it
        self.plan = None             # (what, device id[, song])
        self.expected_until = 0.0
        self.waiting_for_answer = False

    def holds_paused_song(self):
        """A song was paused for this turn, and nothing has changed it."""
        return self.paused_device is not None and self.plan is None

    def wait_for_answer(self):
        self.waiting_for_answer = True
        print("[music] song stays paused until you answer")

    def begin(self):
        self.paused_device = None
        self.plan = None
        self.waiting_for_answer = False

        if not spotify_linked():
            return

        try:
            playback = sp.current_playback()
            device = (playback or {}).get("device") or {}

            if playback and playback.get("is_playing") and is_theo_speaker(device):
                sp.pause_playback(device_id=device["id"])
                self.paused_device = device["id"]
                self.expected_until = 0.0
                print("[spotify] song paused while THEO answers")

        except Exception as error:
            print(f"[spotify] could not pause the song: {error}")

    def finish(self):
        self.waiting_for_answer = False
        plan, self.plan = self.plan, None
        paused, self.paused_device = self.paused_device, None

        if plan is None:
            plan = ("resume", paused) if paused else ("stay paused", None)

        what, device = plan[0], plan[1]

        try:
            if what == "play":
                sp.start_playback(device_id=device, uris=[plan[2]])
                self.expected_until = time.monotonic() + 10.0

            elif what == "resume":
                sp.start_playback(device_id=device)
                self.expected_until = time.monotonic() + 10.0

            elif what in ("next", "previous"):
                if what == "next":
                    sp.next_track(device_id=device)
                else:
                    sp.previous_track(device_id=device)

                # Spotify sometimes leaves a skipped song paused.
                time.sleep(0.5)
                playback = sp.current_playback()
                if not (playback and playback.get("is_playing")):
                    sp.start_playback(device_id=device)
                self.expected_until = time.monotonic() + 10.0

            elif what == "stay paused":
                self.expected_until = 0.0

        except Exception as error:
            print(f"[spotify] {error}")

            if what == "play":
                say(spotify_problem(error), force=True)
            elif what == "resume":
                say("I couldn't carry on with the song.", force=True)
            else:
                say("I couldn't change the track.", force=True)


music_turn = MusicTurn()


def song_and_artist(query):
    """
    Split a request into the song and, when it names one, the artist.

        "Feather by Nujabes"             ->  ("Feather", "Nujabes")
        "track:Feather artist:Nujabes"   ->  ("Feather", "Nujabes")
        "Feather Nujabes"                ->  ("Feather Nujabes", None)
    """
    text = " ".join((query or "").split())
    fields = {}
    for name, value in re.findall(r'\b(track|artist|album):\s*"?(.*?)"?(?=\s+\w+:|$)',
                                  text, re.IGNORECASE):
        fields[name.lower()] = value.strip()
    if fields:
        return fields.get("track", ""), fields.get("artist")
    match = re.match(r"(.+?)\s+by\s+(.+)$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, None


def likeness(heard, actual):
    """0 to 1: how alike two names are, ignoring case, spacing and punctuation."""
    def tidy(name):
        return " ".join(re.findall(r"[a-z0-9]+", (name or "").casefold()))
    return difflib.SequenceMatcher(None, tidy(heard), tidy(actual)).ratio()


def find_track(query):
    """
    The song to play for a request, or None.

    Whatever Spotify finds for the request as it was said is played, exactly
    as before, so requests that already work ("some Radiohead", "focus
    music") are unchanged. Only when that finds nothing does THEO look again
    more loosely, because the words it heard may be misspelt: for "Feather
    by Nujavez" it searches for "Feather" and picks the version whose artist
    sounds closest to "Nujavez", which is Nujabes.
    """
    found = sp.search(q=query, type="track", limit=1)["tracks"]["items"]
    if found:
        return found[0]

    song, artist = song_and_artist(query)
    tries = []
    if artist:
        tries = [(f"{song} {artist}", song, artist), (song, song, artist)]
    else:
        words = song.split()
        for keep in range(len(words) - 1, 0, -1):
            title = " ".join(words[:keep])
            tries.append((title, title, " ".join(words[keep:])))

    for search, title, by in tries:
        if not search.strip():
            continue
        best, best_score = None, 0.0
        for track in sp.search(q=search, type="track", limit=20)["tracks"]["items"]:
            name = re.split(r"\s+[-(\[]", track["name"])[0]
            title_score = likeness(title, name)
            artist_score = max((likeness(by, a["name"]) for a in track["artists"]), default=0.0)
            if title_score >= 0.6 and artist_score >= 0.6 and title_score + artist_score > best_score:
                best, best_score = track, title_score + artist_score
        if best is not None:
            return best

    return None


def play_spotify(query):
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
    try:
        device_id = get_playback_device()
        if device_id is None:
            return THEO_SPEAKER_MISSING

        track = find_track(query)
        if track is None:
            song, artist = song_and_artist(query)
            if artist:
                return f"I couldn't find {song or 'a song'} by {artist} on Spotify."
            return f"I couldn't find a song called {song or query} on Spotify."

        # It starts on THEO's speaker once THEO has said what is playing.
        music_turn.plan = ("play", device_id, track["uri"])
        return f"Playing {track['name']} by {track['artists'][0]['name']}"
    except Exception as e:
        print(f"Spotify error is {e}")
        return spotify_problem(e)


def pause_spotify():
    """Pause the song - or, if it is already paused, carry on with it."""
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
    try:
        if music_turn.paused_device:
            # It was playing when you spoke to THEO, so it is paused
            # already. Now it stays that way.
            music_turn.plan = ("stay paused", None)
            return "Track paused"

        playback = sp.current_playback()
        if playback and playback["is_playing"]:
            sp.pause_playback()
            music_turn.plan = ("stay paused", None)
            return "Track paused"

        device_id = get_playback_device()
        if device_id is None:
            return THEO_SPEAKER_MISSING

        music_turn.plan = ("resume", device_id)
        return "Track resumed"
    except Exception as e:
        print(f"Spotify error is {e}")
        return "Nothing is playing"


def change_track(what, answer):
    if not spotify_linked():
        return SPOTIFY_NOT_LINKED
    try:
        device_id = get_playback_device()
    except Exception as e:
        print(f"Spotify error is {e}")
        return spotify_problem(e)
    if device_id is None:
        return THEO_SPEAKER_MISSING

    # Done once THEO has said so.
    music_turn.plan = (what, device_id)
    return answer


def next_track():
    return change_track("next", "Skipping to the next track")


def previous_track():
    return change_track("previous", "Going back to the previous track")


# ------------------------------------------------------------------
# EYES ON (2026-10-29.1)
#
# "Eyes on" keeps THEO's camera ready for whatever the person shows it.
# Until they say "eyes off", a question that calls for a look ("what does
# this say?", "can you see it now?", "try again") is answered from a
# picture taken the moment they ask - no countdown and no "hold it up" -
# so they can ask about a label, a remote or a screen, one thing after
# another, and THEO can check each step of something it is talking them
# through. Any other question, such as "how do I fix it?", is answered from
# what THEO already saw, and "don't take another picture" is honoured
# (see wants_a_look; until 2026-10-31.1 every question took a picture).
#
# THEO does not film. It takes one picture when it is asked something, so
# nothing is sent anywhere unless the person asks. Between questions the
# ring glows soft white, the light it shows when it takes a picture, so it
# is clear the eyes are on. They go off by themselves after EYES_QUIET
# seconds with no question, and after EYES_LONGEST seconds in any case.
# Reading pictures needs Groq, so without it the eyes stay off and THEO
# says why.
#
# A request for one of THEO's other tools - a timer, a reminder, music,
# the weather - still goes the usual way, and the model can still look, at
# once, with analyze_error.
# ------------------------------------------------------------------

EYES_QUIET = 180.0
EYES_LONGEST = 1200.0

# While the eyes are on, THEO listens this long for the next question
# without "Hey Theo", to give time to hold something up.
EYES_FOLLOW_UP = 30.0

EYES_LOCK = threading.Lock()
eyes = {"since": None, "used": 0.0}

EYES_ON_WORDS = (r"(?:(?:keep|put|turn) )?(?:your )?eyes on(?: (?:this|it|that|here|me|now))?|"
                 r"keep an eye on (?:this|it|that|me)|watch (?:this|me)|open your eyes|"
                 r"(?:turn )?(?:your |the )?camera on|turn on (?:your |the )?camera")
EYES_OFF_WORDS = (r"(?:(?:take|turn) )?(?:your )?eyes off(?: (?:this|it|that|me|now))?|"
                  r"(?:close|shut) your eyes|stop (?:watching|looking)(?: (?:this|it|that|me))?|"
                  r"(?:turn )?(?:your |the )?camera off|turn off (?:your |the )?camera")

EYES_INSTRUCTIONS = (
    "You are THEO, a friendly helper for an older person who is new to technology. Your eyes are "
    "on: this picture was taken by your own camera the moment they spoke, of whatever they are "
    "showing you. Answer what they just said, using the picture and what was said before. If they "
    "are following steps you gave, check whether the picture shows the step done, then give the "
    "next one. If the picture is blurry, too far away or cut off, so that you cannot read it, do not "
    "guess: say so and ask them to hold it closer and keep it still. For medicine, read only what "
    "the label says; never name a pill from how it looks. Reply in one or two short, plain "
    "sentences that will be spoken aloud, with no lists or symbols.")

EYES_TOOLS_NOTE = ("Your eyes are on: the user may be showing you something. When answering "
                   "needs a new look, such as checking a step they say they have done, use "
                   "analyze_error with their question; it takes the picture at once, with no "
                   "countdown. When they ask about what you have already seen, such as how to "
                   "fix it, answer from the conversation without a new picture, and never take "
                   "one when they ask you not to.")


def eyes_are_on():
    """True while THEO's eyes are on. They switch off once unused, or on, for too long."""

    with EYES_LOCK:
        since = eyes["since"]

        if since is None:
            return False

        now = time.monotonic()

        if now - eyes["used"] < EYES_QUIET and now - since < EYES_LONGEST:
            return True

        eyes["since"] = None
        quiet = now - eyes["used"] >= EYES_QUIET

    print("[eyes] eyes off by themselves: " + ("no question for a while" if quiet
                                                else "they have been on a long time"))
    return False


def eyes_on():
    with EYES_LOCK:
        now = time.monotonic()

        if eyes["since"] is None:
            eyes["since"] = now
            print("[eyes] eyes on")

        eyes["used"] = now


def eyes_off():
    """Returns True if they were on."""

    with EYES_LOCK:
        were_on = eyes["since"] is not None
        eyes["since"] = None

    if were_on:
        print("[eyes] eyes off")

    return were_on


def eyes_used():
    """A question while the eyes are on: the quiet time starts again."""

    with EYES_LOCK:
        if eyes["since"] is not None:
            eyes["used"] = time.monotonic()


def eyes_watch():
    """While THEO waits for "Hey Theo", the white glow goes once the eyes go off by themselves."""

    while not STOPPING.wait(2.0):
        with EYES_LOCK:
            were_on = eyes["since"] is not None

        if were_on and not eyes_are_on() and WAITING_FOR_WAKE.is_set():
            show_waiting_ring()


def eyes_command(text):
    """'on' for "eyes on" and the like, 'off' for "eyes off" and the like, None for anything else."""

    addressed = after_name(text)

    if addressed is not None:
        text = addressed

    words = " ".join(re.findall(r"[a-z]+", text.casefold().replace("'", "").replace("’", "")))

    for name, pattern in (("on", EYES_ON_WORDS), ("off", EYES_OFF_WORDS)):
        if re.fullmatch(COMMAND_START + r"(?:" + pattern + r")" + COMMAND_END, words):
            return name

    return None


def after_eyes_on(text):
    """For "Eyes on. What does this say?": what was asked after "eyes on", or None."""

    addressed = after_name(text)

    if addressed is not None:
        text = addressed

    match = re.match(r"\W*(?:" + EYES_ON_WORDS.replace(" ", r"\W+") + r")\W+(\w.*)$",
                     text, re.IGNORECASE)

    if match is None:
        return None

    rest = match.group(1).strip()

    return rest if rest and eyes_command(rest) is None else None


def turn_eyes_on():
    """Eyes on, when pictures can be read. Returns what THEO says."""

    if time.monotonic() < groq_down_until:
        return "I can't see without the internet right now, so my eyes will stay off for now."

    eyes_on()

    return ("My eyes are on. Hold up what you'd like me to see, and ask me anything about it. "
            "Say eyes off when you're done.")


def turn_eyes_off():
    if eyes_off():
        return "Okay, my eyes are off."

    return "My eyes are already off."


def follow_up_window():
    """How long THEO listens for a follow-up without "Hey Theo" (longer while its eyes are on)."""

    return EYES_FOLLOW_UP if eyes_are_on() else CONVERSATION_TIMEOUT


def picture_now():
    """A picture from THEO's camera, taken now, as base64 for the picture reader."""

    image_bytes = node.capture_image()

    print(f"[node] Captured {len(image_bytes):,} bytes")

    # Kept next to this program, so you can open it and see what THEO saw.
    # Each new picture replaces the last.
    try:
        with open(os.path.join(THEO_DIR, "last_picture.jpg"), "wb") as handle:
            handle.write(image_bytes)
        print("[node] Saved as last_picture.jpg")
    except OSError as error:
        print(f"[node] could not save last_picture.jpg: {error}")

    return base64.b64encode(image_bytes).decode("utf-8")


def said_before(messages, exchanges=3):
    """The last few exchanges as plain lines, "Them: ..." and "THEO: ...", for the picture reader."""

    lines = []

    for message in messages[1:-1]:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            continue

        content = message.get("content")

        if isinstance(content, str) and content.strip():
            lines.append(("Them: " if message["role"] == "user" else "THEO: ") + content.strip())

    return lines[-2 * exchanges:]


# Groq's free plan lets the picture reader take in about 7,000 tokens a
# minute, and one of THEO's pictures is about 2,400 of them, so a third
# picture inside a minute is turned away with "Please try again in N s"
# (Aiden's log, 2026-10-31). THEO waits that long, when it is no more than
# VISION_WAIT_LONGEST seconds, and asks once more.
VISION_WAIT_LONGEST = 10.0


def picture_limit_reached(error):
    """True when the picture reader turned a request away for its per-minute limit (HTTP 429)."""

    described = str(error)

    return (getattr(error, "status_code", None) == 429 or "rate_limit" in described or
            "Rate limit" in described)


def vision_wait(error):
    """Seconds to wait before asking the picture reader again, or None when waiting will not help."""

    if not picture_limit_reached(error):
        return None

    match = re.search(r"try again in (?:([0-9]+)m)?([0-9.]+)s", str(error))
    wait = 60 * int(match.group(1) or 0) + float(match.group(2)) if match else 3.0

    return wait + 0.25 if wait <= VISION_WAIT_LONGEST else None


def ask_vision(ask):
    """ask() the picture reader, and once more after a short per-minute limit (see VISION_WAIT_LONGEST)."""

    try:
        return ask()
    except Exception as error:
        wait = vision_wait(error)

        if wait is None:
            raise

        print(f"[vision] the picture reader's per-minute limit is full; asking again in {wait:.1f} s")
        time.sleep(wait)

        return ask()


def picture_problem_reply(error):
    """What THEO says when a picture it took could not be read."""

    if picture_limit_reached(error):
        return ("I've looked at a lot of pictures in the last minute, and the service that "
                "reads them needs a few seconds. Please ask me again in a moment.")

    return ("I took the picture, but I can't reach the service that reads pictures "
            "right now. Please try again in a minute.")


def read_picture(image_data, before, question, notes):
    """What THEO says about a picture just taken with its eyes on (see EYES ON)."""

    text = EYES_INSTRUCTIONS

    if notes:
        text += "\n\nWhat you know about them: " + "; ".join(notes)

    if before:
        text += "\n\nWhat was said just before:\n" + "\n".join(before)

    text += f"\n\nWhat they just said: {question}"

    response = ask_vision(lambda: client.with_options(
        max_retries=0, timeout=GROQ_TIMEOUT).chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": text}]}],
        # As for analyze_screen: two sentences fit easily, and the cap keeps
        # each picture inside Groq's per-minute allowance.
        max_tokens=150
    ))

    return response.choices[0].message.content


def answer_with_eyes(messages, usertext):
    """
    THEO's answer while its eyes are on: a picture taken now, and the
    question asked about it, with what was said before. Returns what
    answer_with_tools returns.
    """

    eyes_used()

    try:
        picture = picture_now()
    except NodeUnavailable as error:
        print(f"[node] {error}")
        return {"reply": error.spoken, "used": ["look"], "problem": None, "stopped": False}

    if stopped_while_answering():
        return {"reply": None, "used": [], "problem": None, "stopped": True}

    try:
        reply = read_picture(picture, said_before(messages), usertext, notes_for(usertext))
    except Exception as error:
        print(f"[vision] could not read the picture: {error}")
        reply = picture_problem_reply(error)

    return {"reply": reply, "used": ["look"], "problem": None, "stopped": False}


def needs_other_tools(usertext):
    """True when a request made with the eyes on is for one of THEO's other tools."""

    return bool(tool_names_for(usertext) - {"analyze_error"})


# With the eyes on, these words call for a new picture: "what do you see?",
# "can you read this?", "what's this?", "is this the right one?", "which
# button?", "where is it?", "how about now?", "try again". Anything else is
# answered from what THEO already saw, and the model can still look when it
# must (EYES_TOOLS_NOTE). Until 2026-10-31.1 every question took a picture:
# in Aiden's log that used up the picture reader's per-minute allowance,
# and a picture was taken even after "don't take another picture".
EYES_LOOK_WORDS = (r"\b(?:see|seeing|look|looking|read|reading|show|showing|watch|check|view)\b|"
                   r"\bwhats (?:this|that|it|on)\b|\bwhat (?:is|are) (?:this|that|these|those|it|on)\b|"
                   r"\bwhat (?:this|that|it) (?:is|says)\b|\bwhat does(?: \w+){1,2} say\b|"
                   r"\b(?:is|are) (?:this|that|these|those)\b|"
                   r"\b(?:is|are) (?:it|they) (?:right|correct|on|off|in|plugged|working)\b|"
                   r"\bwhich (?:one|button|cable|wire|plug|port|key|light|app|icon|setting|side)\b|"
                   r"\bwhat (?:colou?r|number|date|word|words|name|brand|size)\b|\bhow (?:many|much)\b|"
                   r"\bwheres\b|\bwhere (?:is|are|do i|should i)\b|"
                   r"\b(?:do|doing|did) (?:i do )?(?:it|this|that) right\b|"
                   r"\b(?:how|what) about now\b|\band now\b|"
                   r"\btry (?:it |that )?again\b|\bone more (?:time|try)\b")

# ...and these say not to take one, whatever else is said.
EYES_NO_LOOK_WORDS = (r"\b(?:dont|do not)(?: (?:need|have) to)? (?:take|look|snap)\b|"
                      r"\bno need to (?:take|look)\b|\bstop (?:taking|snapping)\b|"
                      r"\bno more (?:pictures|photos|looking)\b|"
                      r"\bwithout (?:a |another |taking a )?(?:picture|photo|looking)\b")


def wants_a_look(usertext):
    """With the eyes on: True when these words call for a new picture (see EYES_LOOK_WORDS)."""

    words = " ".join(re.findall(r"[a-z]+", (usertext or "").casefold().replace("'", "")
                                .replace("’", "")))

    if re.search(EYES_NO_LOOK_WORDS, words):
        return False

    return re.search(EYES_LOOK_WORDS, words) is not None


def wait_for_screen():
    """
    Ask the person to hold their screen up, then photograph it.

    The node has no live preview, so this asks, waits, and takes one shot --
    which is simpler and, for someone holding up a phone, kinder. With THEO's
    eyes on (see EYES ON) they are already showing it, so the picture is
    taken at once.
    """

    if eyes_are_on():
        eyes_used()
        return picture_now()

    say("Please hold your screen up to the camera. I'll take the picture in five seconds.")

    node.set_ring("looking")

    try:
        for _ in range(50):
            time.sleep(0.1)
            if reset_pending():
                raise NodeUnavailable("Picture cancelled for memory reset.",
                                      "The picture was cancelled.")
        image_data = picture_now()
    finally:
        node.set_ring("thinking")

    return image_data


def analyze_screen(image_data, question=None):
    """Send the photograph off to be read: for the error on it, or for the person's question about it."""

    if question:
        asked = (
            "This picture was just taken by THEO's own camera, of something an older user is "
            f"showing it. They asked: {question} Answer simply in 1-2 sentences. If the picture "
            "is blurry, too far away or cut off, so that you cannot read it, do not guess: say "
            "that you can't read it clearly and ask them to hold it closer and keep it still."
        )
    else:
        asked = (
            "This is a screen shown by an older user having a "
            "tech issue. What error or problem do you see? "
            "Give a simple 1-2 sentence description. "
            "If the screen is blurry, too far away or cut off, "
            "so that you cannot read the words on it, do not "
            "guess: say that you can't read it clearly and ask "
            "them to hold it closer and keep it still."
        )

    response = ask_vision(lambda: client.chat.completions.create(
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
                        "text": asked
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
    ))

    return response.choices[0].message.content


def analyze_error(question=None):
    """Photograph the user's screen and describe what is wrong with it, or answer their question about it."""

    try:
        image_data = wait_for_screen()

        # Reading the picture needs Groq. Offline, THEO says so instead of
        # going quiet after asking for the screen.
        try:
            return analyze_screen(image_data, question) if question else analyze_screen(image_data)
        except Exception as error:
            print(f"[vision] could not read the picture: {error}")
            return picture_problem_reply(error)

    except NodeUnavailable as error:
        print(f"[node] {error}")

        # error.spoken is written to be read out loud.
        return error.spoken

    finally:
        # Back to working out the answer.
        node.set_ring("thinking")


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
                if STOPPED_BY_BUTTON.is_set():
                    break
                play_on_node(data, rate)
        else:
            play_on_node(ALARM_TONE, NODE_RATE)


# What THEO said last that the person may have missed: its last answer or
# reminder. "Say that again" says it again.
LAST_SAID = {"text": None}


def announce_alert(text, chime_first=False, epoch=None):
    """
    Finish the announcement, then its chime, as one uninterrupted turn.
    With chime_first, the chime comes first: for a question, so the
    question is the last thing heard before THEO listens for the answer.

    A reminder never talks over someone asking THEO something: it waits
    until they have finished (up to ALERT_PATIENCE seconds). While it plays,
    the listener treats the speech and the chime as THEO's own voice, so
    neither ends up inside your next request.

    Afterwards the ring pulses amber, slowly, until the person next speaks
    to THEO, in case they were out of the room or did not hear it. Stopping
    it with the talk button shows they were there, so then it does not.
    """
    global theo_quiet_since

    if epoch is None:
        epoch = globals().get("PERSONAL_EPOCH", 0)

    waited = 0.0
    while HEARING.is_set() and waited < ALERT_PATIENCE:
        time.sleep(0.2)
        waited += 0.2

    with AUDIO_LOCK:
        if (STOPPING.is_set() or epoch != globals().get("PERSONAL_EPOCH", 0)
                or globals().get("PERSONAL_RESET_BLOCKED", False) or reset_pending()):
            return
        ALERT_PLAYING.set()
        LAST_SAID["text"] = text
        try:
            if chime_first:
                STOPPED_BY_BUTTON.clear()
                try:
                    play_alarm()
                except Exception as error:
                    print(f"[alert] could not sound the chime: {error}")
                if not STOPPED_BY_BUTTON.is_set():
                    say(text, force=True)
            else:
                say(text, force=True)
                if not STOPPED_BY_BUTTON.is_set():
                    try:
                        play_alarm()
                    except Exception as error:
                        print(f"[alert] could not sound the chime: {error}")
        finally:
            theo_quiet_since = time.monotonic()
            ALERT_PLAYING.clear()

        if not STOPPED_BY_BUTTON.is_set():
            REMINDER_LIGHT.set()

            if WAITING_FOR_WAKE.is_set():
                show_ring("reminder")


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

    # --- the model filled in a tool wrongly. ----------------------------
    # Not a connection problem, so THEO does not go offline over it. It is
    # still worth asking again at a steadier setting (see ask_model): the
    # old "trying again would fail the same way" was true of the reminder
    # tool's hours, fixed on 2026-09-22, and is not true of a model slip.
    if model_slipped(error):
        return ("Sorry, I got muddled using one of my tools for that. "
                "Could you ask me again?", False)

    # --- permanent. do not try again. -----------------------------------

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


def model_slipped(error):
    """True when the model filled in a tool wrongly, or reached for a tool it was not offered."""

    low = str(error).lower()
    return "tool_use_failed" in low or "did not match schema" in low


def ask_model(conversation, **request):
    """
    One answer from the language model, asked again only when that can work:
    after a dropped connection, in 5 seconds; after the model fills in a tool
    wrongly, at once and at a steadier setting.

    Returns (response, None), or (None, what to tell the person).
    """

    problem = None

    # 0.7 keeps THEO's answers varied without letting them fall apart (the
    # garbled answers of 2026-10-10 came at 1.0); a retry is steadier still.
    for attempt, temperature in enumerate((0.7, 0.5, 0.2)):
        try:
            return think(messages=conversation, temperature=temperature, **request), None
        except Exception as error:
            problem, worth_retrying = describe_problem(error)
            slipped = model_slipped(error)
            print(f"[problem] {problem}")

            if slipped:
                print(f"[problem] what Groq said: {error}")

            if attempt == 2 or not (worth_retrying or slipped):
                break

            if slipped:
                print("[problem] asking again at a steadier setting")
            else:
                print("[problem] waiting 5 seconds before trying again")
                time.sleep(5)

    return None, problem


def answer_with_tools(messages, usertext, keep_word=False, rounds=3):
    """
    THEO's answer to the question at the end of messages, using every tool
    it needs, round after round, for up to `rounds` rounds of tools.

    Every round offers the tools. Until 2026-10-19.1 only the first did: the
    answer after a tool's result was asked for without them, and when the
    model reached for another tool anyway (Shanghai's weather, 2026-10-19),
    Groq refused the request, the refusal was only printed, and THEO said
    nothing. After the last round, or once a tool has run and the model
    cannot be reached with them, the model is asked for words only.

    keep_word asks again after an answer that promised a reminder, a timer
    or a note and used no tool for it (see promised_without_doing).

    Each tool request and its result are added to messages, as before.
    Returns {"reply", "used", "problem", "stopped"}. reply is None when no
    answer could be had; problem is then what to tell the person, and
    stopped is True if the talk button was pressed while THEO was thinking.
    """

    used = []
    extra = []

    if keep_word:
        extra.append({"role": "system", "content": (
            "Your last answer to this said you would set a reminder or a timer, or "
            "remember something, but you used no tool, so nothing was done. This time "
            "use the tool for each thing you say you will do, then say in one short "
            "sentence what you did.")})

    if eyes_are_on():
        extra.append({"role": "system", "content": EYES_TOOLS_NOTE})

    words_only = {"role": "system", "content": "Answer now in plain words, without using any tool."}

    for round_number in range(rounds + 1):
        last_round = round_number == rounds
        conversation = with_notes(messages, usertext) + extra

        if last_round:
            response, problem = ask_model(conversation + [words_only], tools=tools, tool_choice="none")
        else:
            response, problem = ask_model(conversation, tools=tools, tool_choice="auto")

            # The tools have run: say what they did, rather than nothing.
            if response is None and used:
                last_round = True
                response, problem = ask_model(conversation + [words_only], tools=tools,
                                              tool_choice="none")

        if response is None:
            return {"reply": None, "used": used, "problem": problem, "stopped": False}

        message = response.choices[0].message
        calls = [] if last_round else list(getattr(message, "tool_calls", None) or [])

        # Pressed while THEO was thinking: nothing more is done and nothing is
        # said. (A press during the final answer is dealt with after it.)
        if (round_number == 0 or calls) and stopped_while_answering():
            return {"reply": None, "used": used, "problem": None, "stopped": True}

        if not calls:
            return {"reply": message.content, "used": used, "problem": None, "stopped": False}

        # EVERY tool the model asked for, in order: "play some music and
        # remind me in 20 minutes" is two. (Until 2026-10-16.1 only the first
        # was carried out.) Each result goes back to the model.
        messages.append(plain_message(message, calls))

        for call in calls:
            messages.append({
                "role": "tool",
                "content": run_tool(call),
                "tool_call_id": call.id
            })
            used.append(call.function.name)

    return {"reply": None, "used": used, "problem": None, "stopped": False}


def promised_without_doing(reply, used):
    """
    True when THEO's answer says it set, or will set, a reminder, a timer or
    an alarm, or that it will remember something, while no tool for that ran
    this turn. On 2026-10-19 THEO answered "I'll set a reminder for you in
    two minutes to go check on your cat" and set nothing.
    """

    if not reply or set(used) & {"set_timer", "set_reminder", "set_reminder_at", "remember"}:
        return False

    text = reply.replace("’", "'")
    return bool(re.search(
        r"\b(?:i'll|i will|i've|i have|i'm going to|i am going to|let me)\b[^.?!]{0,60}?"
        r"\b(?:remind|set (?:up )?(?:a |an |the |your )?(?:reminder|timer|alarm)|remember|"
        r"make a note|note that|note it)",
        text, re.IGNORECASE))


def set_timer(time_seconds, label):
    epoch = globals().get("PERSONAL_EPOCH", 0)

    def timer_done():
        try:
            with AUDIO_LOCK:
                with REMINDER_LOCK:
                    if t not in pending_alarms or epoch != globals().get("PERSONAL_EPOCH", 0):
                        return
                announce_alert(f"Time's up! {label}")
        finally:
            with REMINDER_LOCK:
                pending_alarms.discard(t)

    t = threading.Timer(time_seconds, timer_done)
    t.daemon = True
    with REMINDER_LOCK:
        if STOPPING.is_set():
            return "THEO is stopping."
        pending_alarms.add(t)
        t.start()
    return f"{label} set for {time_seconds} seconds."


active_timers = {}  # store timers by label
REMINDER_LOCK = threading.RLock()
pending_alarms = set()


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
        if STOPPING.is_set():
            return "THEO is stopping."
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


# ------------------------------------------------------------------
# REMINDERS AT A TIME OF DAY
#
# "Remind me to take my tablets at 8 every morning." These are kept in
# reminders.json next to this program and loaded again at start-up, so a
# restart forgets nothing. One whose time passed while THEO was off is
# still given if it is less than MISSED_GRACE_MINUTES late.
#
# A medicine reminder chimes, then asks "Did you take it?", and THEO
# listens for the answer without "Hey Theo". If the answer is no, or
# nobody answers, it asks again MEDICINE_AGAIN_MINUTES later, at most
# MEDICINE_ASKS_AGAIN more times. "Yes" is written down, so "what are my
# reminders?" can say when the medicine was last taken.
#
# After any reminder the ring pulses amber, slowly, until the person
# speaks to THEO.
# ------------------------------------------------------------------

REMINDERS_FILE = os.path.join(THEO_DIR, "reminders.json")
MEDICINE_AGAIN_MINUTES = 10
MEDICINE_ASKS_AGAIN = 3
MISSED_GRACE_MINUTES = 120

# How long THEO listens for the answer to "Did you take it?", and how
# fresh the question must be for the listener to still take the answer.
ANSWER_SECONDS = 12.0
ANSWER_FRESH = 30.0

clock_reminders = []          # what reminders.json holds
REMINDER_CLOCK = threading.Condition(REMINDER_LOCK)

pending_answer = None         # the medicine reminder waiting for "yes" or "no"
answer_asked_at = 0.0

PLURAL_MEDICINE = re.compile(
    r"\b(?:tablets|pills|medicines|medications|meds|drops|vitamins|capsules|tabs|injections)\b",
    re.IGNORECASE)


def parse_clock(text):
    """'08:00', '8:00', '20:30', '8:30 pm', '8 am', 'noon' -> (hour, minute); None if it is not a time."""

    text = (text or "").strip().lower()

    if "noon" in text or "midday" in text:
        return (12, 0)

    if "midnight" in text:
        return (0, 0)

    match = re.search(r"(\d{1,2})(?:\s*[:.h]\s*(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?", text)

    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    half = (match.group(3) or "").replace(".", "").replace(" ", "")

    if half == "pm" and hour < 12:
        hour += 12
    elif half == "am" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        return None

    return (hour, minute)


def next_time(hour, minute, after):
    """The first hour:minute after the moment `after`."""

    when = after.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if when <= after:
        when += timedelta(days=1)

    return when


def clock_words(hour, minute):
    """(20, 30) -> '8:30 PM'."""

    return f"{(hour % 12) or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def day_words(when, now):
    """'today', 'tomorrow', 'yesterday' or the day's name, for a moment near now."""

    days = (when.date() - now.date()).days

    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days == -1:
        return "yesterday"

    return "on " + when.strftime("%A")


def save_reminders():
    """Write reminders.json. Call with REMINDER_LOCK held."""

    saved = []

    for item in clock_reminders:
        entry = dict(item)
        entry["next"] = item["next"].isoformat()
        entry["last_taken"] = item["last_taken"].isoformat() if item.get("last_taken") else None
        saved.append(entry)

    try:
        part = REMINDERS_FILE + ".part"

        with open(part, "w") as handle:
            json.dump(saved, handle, indent=1)

        os.replace(part, REMINDERS_FILE)
    except OSError as error:
        print(f"[reminder] could not save reminders.json: {error}")


def load_reminders():
    """Read reminders.json at start-up."""

    try:
        with open(REMINDERS_FILE) as handle:
            saved = json.load(handle)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as error:
        print(f"[reminder] could not read reminders.json ({error}); starting with none")
        return

    now = local_now()
    kept = []

    for entry in saved if isinstance(saved, list) else []:
        try:
            item = {
                "text": " ".join(str(entry["text"]).split()),
                "hour": int(entry["hour"]),
                "minute": int(entry["minute"]),
                "daily": bool(entry.get("daily")),
                "medicine": bool(entry.get("medicine")),
                "again": bool(entry.get("again")),
                "asks": int(entry.get("asks") or 0),
                "next": datetime.fromisoformat(entry["next"]),
                "last_taken": (datetime.fromisoformat(entry["last_taken"])
                               if entry.get("last_taken") else None),
            }
        except (KeyError, TypeError, ValueError, AttributeError):
            print(f"[reminder] skipped a reminder reminders.json could not describe: {entry!r}")
            continue

        if item["next"].tzinfo is None:
            item["next"] = item["next"].replace(tzinfo=THEO_TZ)

        if now - item["next"] > timedelta(minutes=MISSED_GRACE_MINUTES):
            if item["daily"]:
                item["next"] = next_time(item["hour"], item["minute"], now)
            else:
                print(f"[reminder] dropped \"{item['text']}\": its time passed while THEO was off")
                continue

        kept.append(item)

    with REMINDER_LOCK:
        clock_reminders[:] = kept
        save_reminders()

    print(f"[reminder] {len(kept)} reminder(s) at a time of day loaded from reminders.json")


def set_reminder_at(reminder, time_of_day, daily=True, medicine=False):
    """The tool: a reminder at a clock time, once or every day."""

    clock = parse_clock(str(time_of_day))

    if clock is None:
        return f"I couldn't tell what time '{time_of_day}' is. Ask the user for the time."

    text = " ".join(str(reminder or "").split()).rstrip(".")

    if not text:
        return "Ask the user what the reminder is for."

    hour, minute = clock
    now = local_now()

    with REMINDER_LOCK:
        if STOPPING.is_set():
            return "THEO is stopping."

        for item in clock_reminders:
            if (not item["again"] and item["text"].casefold() == text.casefold()
                    and (item["hour"], item["minute"]) == clock):
                item["daily"] = bool(daily)
                item["medicine"] = bool(medicine)
                break
        else:
            item = {"text": text, "hour": hour, "minute": minute, "daily": bool(daily),
                    "medicine": bool(medicine), "again": False, "asks": 0,
                    "next": next_time(hour, minute, now), "last_taken": None}
            clock_reminders.append(item)

        save_reminders()
        REMINDER_CLOCK.notify_all()

    when = clock_words(hour, minute)

    if daily:
        said = f"Reminder set: \"{text}\", every day at {when}, starting {day_words(item['next'], now)}."
    else:
        said = f"Reminder set: \"{text}\", once, at {when} {day_words(item['next'], now)}."

    if medicine:
        said += " As it is medicine, THEO will ask whether it was taken."

    return said


def list_reminders():
    """The tool: every reminder, and when a medicine was last taken."""

    now = local_now()
    lines = []

    with REMINDER_LOCK:
        for item in sorted((i for i in clock_reminders if not i["again"]),
                           key=lambda i: (i["hour"], i["minute"])):
            when = clock_words(item["hour"], item["minute"])

            if item["daily"]:
                line = f"\"{item['text']}\" every day at {when}"
            else:
                line = f"\"{item['text']}\" at {when} {day_words(item['next'], now)}"

            if item["medicine"]:
                taken = item.get("last_taken")

                if taken is not None:
                    line += (f", last taken {day_words(taken, now)} at "
                             f"{clock_words(taken.hour, taken.minute)}")
                else:
                    line += ", not yet taken since it was set"

            lines.append(line)

        for name, timer in active_timers.items():
            minutes = max(1, int(round(getattr(timer, "interval", 60) / 60)))
            lines.append(f"\"{name}\" every {minutes} minute{'s' if minutes != 1 else ''}")

    if not lines:
        return "There are no reminders set."

    return "Reminders: " + "; ".join(lines) + "."


# Words that say nothing about which reminder is meant.
REMINDER_FILLER = {
    "the", "a", "an", "my", "your", "me", "to", "for", "about", "reminder",
    "reminders", "remind", "at", "on", "in", "of", "and", "please", "that",
    "this", "one", "cancel", "stop", "delete", "remove", "o", "clock",
}


def reminder_words(text):
    words = re.findall(r"[a-z0-9]+", (text or "").casefold())
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word
            for word in words if word not in REMINDER_FILLER}


def reminder_likeness(wanted, text):
    """0 to 1: how well the words someone used pick out this reminder."""

    asked = reminder_words(wanted)

    if not asked:
        return 0.0

    return len(asked & reminder_words(text)) / len(asked)


def took_medicine(item, now):
    """'Yes, I took it': write it down, and drop the asking again. Call with REMINDER_LOCK held."""

    for entry in list(clock_reminders):
        if entry["text"].casefold() != item["text"].casefold():
            continue

        if entry["again"]:
            clock_reminders.remove(entry)
        else:
            entry["last_taken"] = now

    save_reminders()
    REMINDER_CLOCK.notify_all()


def cancel_reminder(reminder):
    """
    The tool: cancel a reminder of either kind, every-so-many-minutes or at
    a time of day, from a few of its words ("tablets" finds "take your
    blood pressure tablets") or its time ("the 8 o'clock one").
    """

    wanted = " ".join(str(reminder or "").split())

    with REMINDER_LOCK:
        timer = active_timers.pop(wanted, None)
        if timer is not None:
            timer.cancel()
            return "Reminder cancelled."

        best, best_score = None, 0.0
        clock = parse_clock(wanted)

        for name in active_timers:
            score = reminder_likeness(wanted, name)
            if score > best_score:
                best, best_score = name, score

        for item in clock_reminders:
            if item["again"]:
                continue
            score = reminder_likeness(wanted, item["text"])
            if clock is not None and (item["hour"], item["minute"]) == clock:
                score = max(score, 1.0)
            if score > best_score:
                best, best_score = item, score

        if best is None or best_score < 0.5:
            return "No reminder found with that name."

        if isinstance(best, str):
            active_timers.pop(best).cancel()
            return f"Reminder cancelled: \"{best}\"."

        for entry in list(clock_reminders):
            if entry["text"].casefold() == best["text"].casefold() and (
                    entry is best or entry["again"]):
                clock_reminders.remove(entry)

        save_reminders()
        REMINDER_CLOCK.notify_all()

    return f"Reminder cancelled: \"{best['text']}\" at {clock_words(best['hour'], best['minute'])}."


def give_clock_reminder(item):
    with AUDIO_LOCK:
        if (item.get("_epoch", globals().get("PERSONAL_EPOCH", 0))
                != globals().get("PERSONAL_EPOCH", 0)
                or globals().get("PERSONAL_RESET_BLOCKED", False) or reset_pending()):
            return
        return _give_clock_reminder(item)


def _give_clock_reminder(item):
    """Say a reminder whose time has come. A medicine one asks whether it was taken."""

    global pending_answer, answer_asked_at

    if not item["medicine"]:
        announce_alert(f"Reminder: {item['text']}.")
        return

    question = "Did you take them?" if PLURAL_MEDICINE.search(item["text"]) else "Did you take it?"

    # Asking again is booked before asking, so it happens even if nobody is
    # in the room to answer. "Yes" cancels it.
    with REMINDER_LOCK:
        asks = item["asks"] + 1 if item["again"] else 1

        if asks <= MEDICINE_ASKS_AGAIN:
            clock_reminders.append({
                "text": item["text"], "hour": item["hour"], "minute": item["minute"],
                "daily": False, "medicine": True, "again": True, "asks": asks,
                "next": local_now() + timedelta(minutes=MEDICINE_AGAIN_MINUTES),
                "last_taken": None})
            save_reminders()
            REMINDER_CLOCK.notify_all()

        pending_answer = dict(item, asks=asks)

    announce_alert(f"Reminder: {item['text']}. {question}", chime_first=True)

    if not STOPPING.is_set():
        answer_asked_at = time.monotonic()
        ANSWER_WANTED.set()


def medicine_not_answered():
    """Nobody answered "Did you take it?". The asking again is already booked."""

    global pending_answer

    with REMINDER_LOCK:
        pending_answer = None


def answer_medicine(text):
    """
    If a medicine reminder is waiting for its answer and `text` is a yes or
    a no, deal with it and return what THEO says. Otherwise None: it is an
    ordinary request, and the reminder asks again later.
    """

    global pending_answer

    with REMINDER_LOCK:
        item = pending_answer

    if item is None:
        return None

    answer = yes_or_no(text)

    with REMINDER_LOCK:
        pending_answer = None
        ANSWER_WANTED.clear()

        if answer == "yes":
            took_medicine(item, local_now())

    if answer is None:
        return None

    it = "them" if PLURAL_MEDICINE.search(item["text"]) else "it"

    if answer == "yes":
        if item["daily"] or item["again"]:
            with REMINDER_LOCK:
                daily = next((entry for entry in clock_reminders
                              if entry["daily"] and not entry["again"]
                              and entry["text"].casefold() == item["text"].casefold()), None)
            if daily is not None:
                return (f"Well done. I'll remind you again {day_words(daily['next'], local_now())} "
                        f"at {clock_words(daily['hour'], daily['minute'])}.")
        return "Well done."

    if item["asks"] <= MEDICINE_ASKS_AGAIN:
        return f"Okay. I'll remind you again in {MEDICINE_AGAIN_MINUTES} minutes."

    return f"Okay. I won't ask again this time, so please remember to take {it}."


NOT_TAKEN = re.compile(
    r"\b(?:not yet|haven'?t|have not|didn'?t|did not|not now|neva|never|later|forgot|nuh yet)\b",
    re.IGNORECASE)
TAKEN = re.compile(
    r"\b(?:already|aready|took|taken|tek|done|i did|i have|i've had|had (?:it|them))\b",
    re.IGNORECASE)
PLAIN_YES = re.compile(
    r"\b(?:yes|yeah|yep|yup|yah|ya|yea|sure|of course|correct|uh huh|mm hmm|mhm)\b", re.IGNORECASE)
PLAIN_NO = re.compile(r"\b(?:no|nope|nah|nuh)\b", re.IGNORECASE)


def yes_or_no(text):
    """'yes', 'no' or None, for an answer to "Did you take it?"."""

    text = " ".join((text or "").replace("-", " ").split())

    if NOT_TAKEN.search(text):
        return "no"
    if TAKEN.search(text):
        return "yes"
    if PLAIN_NO.search(text):
        return "no"
    if PLAIN_YES.search(text):
        return "yes"

    return None


def take_due_reminder(now):
    """
    The reminder whose time has come, or None. A daily one moves on to its
    next time, anything else leaves the list. Call with REMINDER_LOCK held.
    """

    due = [item for item in clock_reminders if item["next"] <= now]

    if not due:
        return None

    item = min(due, key=lambda entry: entry["next"])

    if item["daily"]:
        item["next"] = next_time(item["hour"], item["minute"], max(now, item["next"]))
    else:
        clock_reminders.remove(item)

    save_reminders()

    return dict(item)


def reminder_clock():
    """
    Runs beside THEO for as long as it runs (started near the end of this
    program): gives each reminder at a time of day when its time comes.
    """

    while not STOPPING.is_set():
        with REMINDER_CLOCK:
            now = local_now()
            item = take_due_reminder(now)

            if item is not None:
                item["_epoch"] = globals().get("PERSONAL_EPOCH", 0)

            if item is None:
                soonest = min((entry["next"] for entry in clock_reminders), default=None)
                wait = 30.0 if soonest is None else min(30.0, max(0.2, (soonest - now).total_seconds()))
                REMINDER_CLOCK.wait(wait)
                continue

        try:
            give_clock_reminder(item)
        except Exception as error:
            print(f"[reminder] could not give the reminder \"{item['text']}\": {error}")


# ------------------------------------------------------------------
# WHAT THEO REMEMBERS ABOUT THE PERSON
#
# "Remember my granddaughter is Keisha", "my phone is a Samsung". The
# notes are kept in theo_memory.json next to this program, as many as
# the person likes; "what do you remember about me?" and "forget that"
# work too.
#
# THEO does NOT read the whole file to the model, at start-up or ever.
# Each question carries only the notes that fit it: always the ones about
# the person's family and their phone or other devices (at most
# MEMORY_ALWAYS), plus the ones whose words match the question (at most
# MEMORY_MATCHING), all together at most MEMORY_LETTERS letters - about
# 225 tokens - however many notes there are. The model can look further
# with the recall tool. So a big memory costs no more per question than
# a small one.
#
# The conversation is trimmed the same way: each question carries the
# last CONVERSATION_KEEP exchanges, whole, so a tool's request and its
# result always stay together. Until 2026-10-16.2 THEO sent every
# exchange since it started, with every question.
# ------------------------------------------------------------------

MEMORY_FILE = os.path.join(THEO_DIR, "theo_memory.json")
MEMORY_KINDS = ("family", "phone and devices", "health", "likes", "home", "other")
MEMORY_ALWAYS_KINDS = ("family", "phone and devices")
MEMORY_ALWAYS = 8
MEMORY_MATCHING = 6
MEMORY_LETTERS = 900
CONVERSATION_KEEP = 10

memory_notes = []
MEMORY_LOCK = threading.RLock()

# Words that say nothing about what a note is about.
MEMORY_FILLER = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did",
    "my", "your", "her", "his", "their", "our", "me", "i", "you", "she", "he", "they",
    "we", "it", "its", "them", "him", "that", "this", "these", "those", "what", "who",
    "how", "when", "where", "which", "why", "can", "could", "would", "will", "should",
    "please", "theo", "remember", "know", "tell", "about", "called", "named", "name",
    "also", "just", "very", "really", "some", "any", "there", "here", "now", "today",
    "yes", "no", "not", "so", "as", "if", "then", "than", "from", "up", "out", "get",
    "like", "want", "need", "help", "use", "using", "used", "one", "thing", "things",
}

# Different words for the same thing, so a question finds the note.
MEMORY_SAME = {
    "granddaughter": "grandchild", "grandson": "grandchild", "grandkid": "grandchild",
    "grandchildren": "grandchild", "grandbaby": "grandchild",
    "cellphone": "phone", "mobile": "phone", "cell": "phone", "smartphone": "phone",
    "iphone": "phone", "samsung": "phone", "android": "phone",
    "tv": "television", "telly": "television",
    "ipad": "tablet", "laptop": "computer", "pc": "computer",
    "doctor": "doctor", "dr": "doctor", "physician": "doctor",
    "pill": "medicine", "meds": "medicine", "medication": "medicine", "medicines": "medicine",
    "wife": "spouse", "husband": "spouse", "partner": "spouse",
    "whatsapp": "whatsapp", "wifi": "wifi", "wi": "wifi", "internet": "wifi",
    "music": "music", "song": "music", "songs": "music", "reggae": "music",
    "dancehall": "music", "gospel": "music", "jazz": "music", "ska": "music",
    "calypso": "music", "soca": "music", "spotify": "music",
}


def memory_words(text):
    """The words of a note or question that say what it is about, made comparable."""

    words = re.findall(r"[a-z0-9]+", (text or "").casefold().replace("'", "").replace("’", ""))
    kept = set()

    for word in words:
        if word in MEMORY_FILLER:
            continue

        word = MEMORY_SAME.get(word, word)

        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = MEMORY_SAME.get(word[:-1], word[:-1])

        kept.add(word)

    return kept


def save_memory():
    """Write theo_memory.json. Call with MEMORY_LOCK held."""

    try:
        part = MEMORY_FILE + ".part"

        with open(part, "w") as handle:
            json.dump(memory_notes, handle, indent=1)

        os.replace(part, MEMORY_FILE)
    except OSError as error:
        print(f"[memory] could not save theo_memory.json: {error}")


def load_memory():
    """Read theo_memory.json at start-up."""

    try:
        with open(MEMORY_FILE) as handle:
            saved = json.load(handle)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as error:
        print(f"[memory] could not read theo_memory.json ({error}); THEO remembers nothing for now")
        return

    kept = []

    for entry in saved if isinstance(saved, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"].strip():
            kind = entry.get("kind") if entry.get("kind") in MEMORY_KINDS else "other"
            kept.append({"text": " ".join(entry["text"].split()), "kind": kind,
                         "added": str(entry.get("added") or "")})

    with MEMORY_LOCK:
        memory_notes[:] = kept

    print(f"[memory] {len(kept)} thing(s) remembered about the person (theo_memory.json)")


def note_likeness(words, text):
    """0 to 1: how much of what was asked about a note is in it."""

    if not words:
        return 0.0

    return len(words & memory_words(text)) / len(words)


def remember(fact, about="other", replaces=None):
    """The tool: keep a note about the person, for next time."""

    text = " ".join(str(fact or "").split()).strip()

    if not text:
        return "Ask the user what to remember."

    kind = about if about in MEMORY_KINDS else "other"

    with MEMORY_LOCK:
        dropped = []

        if replaces:
            wanted = memory_words(replaces)

            for note in list(memory_notes):
                if wanted and note_likeness(wanted, note["text"]) >= 0.5:
                    memory_notes.remove(note)
                    dropped.append(note["text"])

        for note in memory_notes:
            if note["text"].casefold().rstrip(".") == text.casefold().rstrip("."):
                return "That is remembered already."

        memory_notes.append({"text": text, "kind": kind,
                             "added": local_now().strftime("%Y-%m-%d %H:%M")})
        save_memory()

    print(f"[memory] remembered: {text}" + (f" (instead of: {'; '.join(dropped)})" if dropped else ""))

    return "Remembered." + (f" It replaces: {'; '.join(dropped)}." if dropped else "")


def recall(topic=""):
    """The tool: what THEO remembers, all of it or about one thing."""

    with MEMORY_LOCK:
        notes = list(memory_notes)

    if not notes:
        return "THEO does not remember anything about the user yet."

    wanted = memory_words(topic)

    if wanted:
        found = sorted((n for n in notes if note_likeness(wanted, n["text"]) > 0),
                       key=lambda n: note_likeness(wanted, n["text"]), reverse=True)

        if not found:
            return f"Nothing remembered about {topic}."

        return "Remembered: " + "; ".join(n["text"] for n in found[:10]) + "."

    # Everything: the people and devices first, then the newest.
    ordered = ([n for n in notes if n["kind"] in MEMORY_ALWAYS_KINDS] +
               [n for n in reversed(notes) if n["kind"] not in MEMORY_ALWAYS_KINDS])
    shown = ordered[:12]
    said = "Remembered: " + "; ".join(n["text"] for n in shown) + "."

    if len(notes) > len(shown):
        said += f" And {len(notes) - len(shown)} more things; ask about a topic to hear them."

    return said


def forget(what):
    """The tool: forget a note. 'that' means the last one remembered; 'everything' means all."""

    words = " ".join(str(what or "").split())
    plain = words.casefold().strip(" .!")

    with MEMORY_LOCK:
        if not memory_notes:
            return "There is nothing remembered to forget."

        if plain in ("everything", "all", "all of it", "everything about me"):
            count = len(memory_notes)
            memory_notes.clear()
            save_memory()
            print("[memory] forgot everything")
            return f"Forgot all {count} things."

        if plain in ("", "that", "it", "this", "the last thing", "what i just said"):
            note = memory_notes.pop()
        else:
            wanted = memory_words(words)
            scored = [(note_likeness(wanted, n["text"]), i) for i, n in enumerate(memory_notes)]
            score, index = max(scored) if scored else (0.0, -1)

            if score < 0.5:
                return f"Nothing remembered matches '{words}'."

            note = memory_notes.pop(index)

        save_memory()

    print(f"[memory] forgot: {note['text']}")

    return f"Forgot: {note['text']}."


def notes_for(question):
    """
    The notes that go with this question: the family and the devices, then
    the ones whose words match it, up to MEMORY_LETTERS letters in all.
    """

    with MEMORY_LOCK:
        notes = list(memory_notes)

    if not notes:
        return []

    chosen = [n for n in notes if n["kind"] in MEMORY_ALWAYS_KINDS][-MEMORY_ALWAYS:]

    wanted = memory_words(question)

    if wanted:
        matching = sorted(((note_likeness(wanted, n["text"]), i) for i, n in enumerate(notes)
                           if n not in chosen),
                          reverse=True)
        chosen += [notes[i] for score, i in matching[:MEMORY_MATCHING] if score > 0]

    sent, letters = [], 0

    for note in chosen:
        if letters + len(note["text"]) + 2 > MEMORY_LETTERS:
            continue

        sent.append(note["text"])
        letters += len(note["text"]) + 2

    return sent


# How the notes begin when they go with a question (offline_request knows them by it).
NOTES_HEADING = "Things you remember about the user."


def with_notes(messages, question):
    """
    What is sent for one question: the conversation, with the notes that fit
    the question just after THEO's instructions. Without notes, unchanged.
    """

    notes = notes_for(question)

    if not notes:
        return list(messages)

    with MEMORY_LOCK:
        others = len(memory_notes) - len(notes)

    text = (NOTES_HEADING + " Use them when they help, and don't list them "
            "unless asked:\n- " + "\n- ".join(notes))

    if others > 0:
        text += f"\n({others} more notes; the recall tool finds them.)"

    return list(messages[:1]) + [{"role": "system", "content": text}] + list(messages[1:])


def keep_recent(messages):
    """
    Trim the conversation, in place, to THEO's instructions and the last
    CONVERSATION_KEEP exchanges. An exchange starts with something the
    person said and runs to the next, so a tool's request and its result
    are always kept or dropped together.
    """

    starts = [i for i, message in enumerate(messages)
              if i > 0 and isinstance(message, dict) and message.get("role") == "user"]

    if len(starts) <= CONVERSATION_KEEP:
        return

    del messages[1:starts[-CONVERSATION_KEEP]]


# Capitalised words in the notes - "Keisha", "Samsung", "Half Way Tree" -
# join the listening hint, so they are heard right. Days and months are
# written with capitals too, but are no help there.
NAME_IN_NOTE = re.compile(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*")
NOT_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mondays", "tuesdays", "wednesdays", "thursdays", "fridays", "saturdays", "sundays",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "i", "theo",
}


def remembered_names():
    """Names in the notes, newest first, for the listening hint."""

    with MEMORY_LOCK:
        notes = list(reversed(memory_notes))

    names = []

    for note in notes:
        for found in NAME_IN_NOTE.findall(note["text"]):
            # "Her", "Your", "She" at the start of a note are not names.
            words = found.split()

            while words and words[0].casefold() in MEMORY_FILLER:
                words.pop(0)

            name = " ".join(words)

            if name and name.casefold() not in NOT_NAMES and \
                    name.casefold() not in (n.casefold() for n in names):
                names.append(name)

    return names[:30]


PERSONAL_EPOCH = 0
PERSONAL_RESET_BLOCKED = False
PERSONAL_SETUP = None
reset_handled = 0


def reset_pending():
    listener = globals().get("ears")
    mic = getattr(listener, "mic", None)
    return getattr(mic, "reset_requests", 0) > globals().get("reset_handled", 0)


def reset_personal_state():
    global reset_handled, PERSONAL_EPOCH, PERSONAL_RESET_BLOCKED
    global pending_answer, answer_asked_at, last_heard_at, answering_from

    reset_handled = getattr(getattr(globals().get("ears"), "mic", None), "reset_requests", 0)
    PERSONAL_RESET_BLOCKED = True

    with AUDIO_LOCK, MEMORY_LOCK, REMINDER_CLOCK:
        PERSONAL_EPOCH += 1
        for timer in list(pending_alarms) + list(active_timers.values()):
            timer.cancel()
        pending_alarms.clear()
        active_timers.clear()
        clock_reminders.clear()
        memory_notes.clear()
        del messages[1:]
        LAST_SAID["text"] = None
        pending_answer = None
        answer_asked_at = 0.0
        last_heard_at = None
        answering_from = None
        ANSWER_WANTED.clear()
        REMINDER_LIGHT.clear()
        eyes_off()
        voice_settings.update(speed=1.0, loudness=1.0)
        REMINDER_CLOCK.notify_all()
        try:
            if music_on_speaker(node_audio_status()):
                music_turn.begin()
        except Exception as error:
            print(f"[reset] could not pause music: {error}")
        music_turn.paused_device = None
        music_turn.plan = None
        music_turn.waiting_for_answer = False
        music_turn.expected_until = 0.0
        stop_speaking()
        if talk_button is not None:
            talk_button.mark()
        mic = getattr(globals().get("ears"), "mic", None)
        if mic is not None:
            mic.drain()
        try:
            PERSONAL_SETUP.reset()
        except OSError as error:
            print(f"[reset] personal memory reset incomplete: {error}")
            say("I couldn't finish clearing my memory. Please hold the reset button again to retry.",
                force=True)
            return False
        PERSONAL_RESET_BLOCKED = False
        STOPPED_BY_BUTTON.clear()
        print("[reset] personal memory cleared; starting spoken setup")
        return True


def setup_answer(text):
    with MEMORY_LOCK:
        try:
            reply, notes = PERSONAL_SETUP.answer(
                text, memory_notes, local_now().strftime("%Y-%m-%d %H:%M"))
            memory_notes[:] = notes
            return reply
        except OSError as error:
            print(f"[setup] could not save the answer: {error}")
            load_memory()
            return "I couldn't save that just now. Please try again, or we can finish this later."


# ------------------------------------------------------------------
# THEO ON WHATSAPP (2026-10-17.1)
#
# theo_whatsapp.py runs on the Uno Q, outside App Lab, as the service
# theo-whatsapp. It answers WhatsApp itself. What has to happen here - a
# message from the family said out loud, a reminder they set - it keeps in
# a small queue, and THEO collects it from there every few seconds,
# between turns, and does it with its own voice and reminder code. So this
# program stays the only one that uses the speaker.
#
# Each time THEO checks in it also says how it is (online, the board,
# its reminders), so "How is THEO?" on WhatsApp gets a real answer.
#
# Switched on by one line in key.env, the same line that is in the
# gateway's ~/.theo_whatsapp/whatsapp.env (install_theo_whatsapp.sh prints
# it):
#
#     THEO_LINK_TOKEN=...
#
# THEO reaches the gateway on the Uno Q through App Lab's container
# gateway, as it reaches Ollama. If the gateway runs somewhere else, put
# its address in key.env as THEO_LINK_URL=http://<address>:5055
# ------------------------------------------------------------------

THEO_UPDATE = "2026-10-31.1"

WHATSAPP_LINK_PORT = 5055
WHATSAPP_LINK_EVERY = 3.0        # seconds between check-ins
WHATSAPP_LINK_TIMEOUT = 5.0

# A message waits at most this long for THEO to finish what it is doing
# (a conversation, a song's question, a medicine reminder's "Did you take
# it?"). After that it is said anyway, as a reminder would be.
WHATSAPP_PATIENCE = 60.0

# The jobs already done, so one is never done twice if the gateway asks
# again because THEO's "done" did not reach it.
whatsapp_done = {}

# When someone last spoke to THEO, for "How is THEO?".
last_heard_at = None


def whatsapp_link_token():
    return (os.environ.get("THEO_LINK_TOKEN") or "").strip()


def whatsapp_link_address():
    """Where the WhatsApp gateway listens: THEO_LINK_URL, or port 5055 on the Uno Q."""

    configured = (os.environ.get("THEO_LINK_URL") or "").strip().rstrip("/")

    if configured:
        return configured

    return f"http://{container_gateway() or 'localhost'}:{WHATSAPP_LINK_PORT}"


def theo_status():
    """How THEO is, for the WhatsApp gateway: sent each time THEO checks in."""

    with REMINDER_LOCK:
        reminders = [{
            "text": item["text"],
            "when": clock_words(item["hour"], item["minute"]),
            "daily": item["daily"],
            "medicine": item["medicine"],
            "next": item["next"].isoformat(),
            "last_taken": item["last_taken"].isoformat() if item.get("last_taken") else None,
        } for item in sorted((i for i in clock_reminders if not i["again"]),
                             key=lambda i: (i["hour"], i["minute"]))]

    mic = getattr(globals().get("ears"), "mic", None)

    return {
        "update": THEO_UPDATE,
        "time": local_now().isoformat(),
        "online": not brain_offline,
        "voice": "the offline voice" if getattr(say, "offline_only", False) else "ElevenLabs",
        "board": NODE_IP,
        "board_connected": bool(getattr(mic, "connected", False)),
        "last_heard": last_heard_at.isoformat() if last_heard_at else None,
        "between_turns": WAITING_FOR_WAKE.is_set(),
        "reminders": reminders,
    }


def between_turns():
    """True while THEO waits for "Hey Theo" with nothing else going on."""

    setup = globals().get("PERSONAL_SETUP")
    return (not globals().get("PERSONAL_RESET_BLOCKED", False)
            and not (setup is not None and setup.active)
            and WAITING_FOR_WAKE.is_set() and not HEARING.is_set() and not THEO_TALKING.is_set()
            and not ALERT_PLAYING.is_set() and not ANSWER_WANTED.is_set())


def wait_until_between_turns(limit=WHATSAPP_PATIENCE):
    waited = 0.0

    while not between_turns() and waited < limit and not STOPPING.is_set():
        time.sleep(0.5)
        waited += 0.5


def do_whatsapp_job(job):
    """One job from WhatsApp, done with THEO's own voice or reminder code: (done, what happened)."""

    kind = job.get("kind")
    args = job.get("args") or {}
    name = " ".join(str(job.get("name") or "").split()) or "the family"

    if STOPPING.is_set():
        return False, "THEO is stopping."

    if kind == "say":
        text = " ".join(str(args.get("text") or "").split())

        if not text:
            return False, "The message was empty."

        # Said like a reminder: chime first, then the message, and the ring
        # pulses amber afterwards in case nobody was in the room. "Say that
        # again" repeats it.
        announce_alert(f"A message from {name}: {text}", chime_first=True)

        if STOPPED_BY_BUTTON.is_set():
            return True, "stopped"

        return True, "said"

    if kind == "remind":
        result = set_reminder_at(args.get("reminder"), args.get("time"),
                                 yes_from_model(args.get("daily"), False),
                                 yes_from_model(args.get("medicine"), False))
        return result.startswith("Reminder set"), result

    return False, f"This THEO doesn't know how to do \"{kind}\". Its main.py may need updating."


def finish_whatsapp_job(address, headers, job):
    job_id = str(job.get("id") or "")

    if job_id in whatsapp_done:
        done, result = whatsapp_done[job_id]
    else:
        wait_until_between_turns()

        try:
            done, result = do_whatsapp_job(job)
        except Exception as error:
            done, result = False, f"That did not work ({error})."

        whatsapp_done[job_id] = (done, result)

        while len(whatsapp_done) > 100:
            whatsapp_done.pop(next(iter(whatsapp_done)))

        print(f"[whatsapp] {job.get('kind')} from {job.get('name')}: {result}")

    try:
        requests.post(address + "/theo/done", json={"id": job_id, "ok": done, "result": result},
                      headers=headers, timeout=WHATSAPP_LINK_TIMEOUT)
    except Exception as error:
        print(f"[whatsapp] could not tell the WhatsApp gateway that was done ({error}); "
              "it will ask again, and THEO won't do it twice")


def whatsapp_link():
    """
    Runs beside THEO for as long as it runs (started near the end of this
    program), once key.env has THEO_LINK_TOKEN: checks in with the WhatsApp
    gateway every WHATSAPP_LINK_EVERY seconds and does the jobs it hands back.
    """

    address = whatsapp_link_address()
    headers = {"Authorization": "Bearer " + whatsapp_link_token()}
    state = None

    print(f"[whatsapp] on: checking for messages and reminders from WhatsApp at {address}")

    while not STOPPING.is_set():
        time.sleep(WHATSAPP_LINK_EVERY)

        try:
            reply = requests.post(address + "/theo/poll", json={"status": theo_status()},
                                  headers=headers, timeout=WHATSAPP_LINK_TIMEOUT)
        except Exception as error:
            if state != "unreachable":
                print(f"[whatsapp] can't reach the WhatsApp gateway at {address} ({error}). Is it "
                      "running? On the Uno Q: journalctl -u theo-whatsapp -f")
            state = "unreachable"
            continue

        if reply.status_code == 401:
            if state != "refused":
                print("[whatsapp] the WhatsApp gateway turned THEO down: THEO_LINK_TOKEN in key.env "
                      "must be the same as in ~/.theo_whatsapp/whatsapp.env on the Uno Q")
            state = "refused"
            continue

        try:
            reply.raise_for_status()
            jobs = reply.json().get("jobs") or []
        except Exception as error:
            if state != "confused":
                print(f"[whatsapp] the WhatsApp gateway gave an answer THEO can't read ({error})")
            state = "confused"
            continue

        if state != "connected":
            print("[whatsapp] connected to the WhatsApp gateway")
            state = "connected"

        for job in jobs:
            finish_whatsapp_job(address, headers, job)


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


def one_reply(text):
    """
    The language model sometimes sends two versions of its answer run
    together with no space between them, so THEO answered twice:
    "How can I help?Hi again, Jorge! What would you like to do today?".
    The first version is kept. A full stop, question or exclamation mark
    after a small letter or number, with a capitalised word straight after
    it and no space, only happens where two answers were glued together.
    """
    if not text:
        return text
    glued = re.compile(r"(?<=[a-z0-9][.!?])(?=[A-Z][a-z])|(?<=\))(?=\()")
    return glued.split(text.strip(), maxsplit=1)[0].strip()


def asks_question(text):
    """True when THEO's answer ends by asking the person something."""
    return bool(text) and text.rstrip().rstrip("\"'”’)").endswith("?")


# The model's own working notes, when they leak into the answer. The end of
# the 2026-10-10 session spoke one aloud: "We need to respond concisely, 2
# sentences max."
MODEL_NOTES = re.compile(
    r"\b(?:we|i) (?:need|should|must|have) to (?:respond|answer|reply)\b"
    r"|\bsentences? max(?:imum)?\b"
    r"|\brespond concisely\b"
    r"|\bthe user (?:said|says|asked|asks|wants|is asking)\b",
    re.IGNORECASE)


def plain_spacing(text):
    """
    The text with invisible characters taken out (zero-width spaces and
    other format marks) and every kind of space - no-break, narrow and so
    on - made an ordinary space. The garbled answers in the 2026-10-10
    session were built from exactly those, e.g. "**" + zero-width space +
    "**" between rows of no-break spaces.
    """
    return "".join(" " if character.isspace() else character
                   for character in text
                   if unicodedata.category(character) != "Cf")


def looks_garbled(text):
    """
    True when an answer is not fit to say out loud and should be asked for
    again: (nearly) no words, ellipses one after another, runs of stars or
    blank space, or the model's own working notes. Three answers like that
    reached the speaker at the end of the 2026-10-10 session.
    """
    if not text:
        return True
    text = plain_spacing(text)
    if not text.strip():
        return True
    if sum(1 for character in text if character.isalpha()) < 4:
        return True
    if text.count("…") + len(re.findall(r"\.{3,}", text)) >= 3:
        return True
    if re.search(r"(?:…|\.{3})\s*(?:…|\.{3})", text):
        return True
    if re.search(r"\*{3,}| {6,}", text):
        return True
    return bool(MODEL_NOTES.search(text))


def speakable(text):
    """
    What the voice should actually say. The answer is printed as written,
    but stars, hashes and underscores from formatting, emoji and other
    picture symbols are left out of the speech, and blank space is tidied.
    A degree sign stays: "30°C" is still read as a temperature.
    """
    text = re.sub(r"[*_`#~]+", "", plain_spacing(text or ""))
    text = "".join(character for character in text
                   if character == "°" or unicodedata.category(character) != "So")
    return re.sub(r"\s+", " ", text).strip()


def ask_again(conversation):
    """One more try at an answer that came out garbled: no tools, steadier sampling."""
    try:
        again = think(
            messages=conversation,
            temperature=0.3
        )
        return again.choices[0].message.content or ""
    except Exception as error:
        print(f"[reply] could not ask again: {error}")
        return ""


def yes_from_model(value, default):
    """A true or false from the model, which sometimes writes it as the word: "false" means False."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"):
        return True

    if isinstance(value, str) and value.strip().lower() in ("false", "no", "0"):
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    return default


def run_tool(tool_call):
    """
    Carry out one tool the model asked for, and return its result in words
    for the model. A tool that fails says so in its result, so the other
    tools in the same answer still run and THEO can still reply.
    """

    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except ValueError:
        args = {}

    if not isinstance(args, dict):
        args = {}

    try:
        if name == "get_time":
            return get_time()
        if name == "get_weather":
            return get_weather(args["city"])
        if name == "play_spotify":
            return play_spotify(args["query"])
        if name == "pause_spotify":
            return pause_spotify()
        if name == "next_track":
            return next_track()
        if name == "previous_track":
            return previous_track()
        if name == "analyze_error":
            return analyze_error(args.get("question"))
        if name == "set_timer":
            return set_timer(args["time"], args["label"])
        if name == "set_reminder":
            return set_reminder(args["reminder"], args["minutes"])
        if name == "set_reminder_at":
            return set_reminder_at(args["reminder"], args["time"],
                                   yes_from_model(args.get("daily"), True),
                                   yes_from_model(args.get("medicine"), False))
        if name == "list_reminders":
            return list_reminders()
        if name == "cancel_reminder":
            return cancel_reminder(args["reminder"])
        if name == "remember":
            return remember(args["fact"], args.get("about", "other"), args.get("replaces"))
        if name == "recall":
            return recall(args.get("topic", ""))
        if name == "forget":
            return forget(args.get("what", ""))
        if name == "web_search":
            return web_search(args["query"])
    except Exception as error:
        print(f"[tool] {name} did not work: {error!r}")
        return f"That did not work ({error})."

    return "I don't know how to use that tool."


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
                        "description": "The name of the city the user is in. E.g- Kingston, London, etc"
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
            "description": "Uses THEO's own camera: asks the user to hold their screen up, takes the picture and reads the error on it. ALWAYS use this as soon as the user mentions an error, a message on a screen, or asks you to look at, check or read something. The user cannot send or upload pictures; this tool is the only way to see their screen. With your eyes on, the picture is taken at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What the user wants to know about what they are showing, in their words, e.g. 'what does this label say'. Leave it out for an error on a screen."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Sounds the alarm and says the label once, after the given number of seconds. Use it for 'set a timer for 5 minutes' and for a single reminder after a while, e.g. 'remind me in 2 minutes to check on the cat' (time 120, label 'check on the cat').",
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
            "description": "Reminds the user over and over, every so many minutes, until they cancel it: only for repeating reminders such as 'every 30 minutes'. For a single reminder after a while ('in 10 minutes') use set_timer; for a time of day ('at 8', 'every morning at 7') use set_reminder_at instead.",
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
            "name": "set_reminder_at",
            "description": "Reminds the user at a time of day, once or every day, e.g. 'remind me to take my tablets at 8 every morning'. Set medicine to true for medicine: THEO then asks whether they took it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder": {
                        "type": "string",
                        "description": "What to remind them about, as you will say it to them, e.g. 'take your blood pressure tablets'."
                    },
                    "time": {
                        "type": "string",
                        "description": "The time of day on a 24-hour clock, HH:MM. 8 in the morning is 08:00, half past 8 at night is 20:30."
                    },
                    "daily": {
                        "type": "boolean",
                        "description": "true to remind every day at this time, false for only once."
                    },
                    "medicine": {
                        "type": "boolean",
                        "description": "true if it is about taking medicine."
                    }
                },
                "required": ["reminder", "time", "daily", "medicine"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Keeps a note about the user for next time. Use it when they ask you to remember something, and when they tell you something about themselves that will help you help them later: family names, their phone or other devices, health needs, what they like.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The note, as you would say it back to them, e.g. 'your granddaughter is called Keisha' or 'your phone is a Samsung Galaxy A15'."
                    },
                    "about": {
                        "type": "string",
                        "enum": ["family", "phone and devices", "health", "likes", "home", "other"],
                        "description": "What the note is about."
                    },
                    "replaces": {
                        "type": "string",
                        "description": "Only if it replaces an older note (a new phone, say): words from the old note."
                    }
                },
                "required": ["fact", "about"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Looks up what you remember about the user. Use it for 'what do you remember about me?', or to find a note about something that is not in the notes you were given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "What to look up, e.g. 'grandchildren'. Leave it empty for everything."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Forgets a note when the user asks. 'Forget that' means the last thing remembered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "description": "Words from the note to forget, or 'that' for the last one, or 'everything' only if they clearly ask to forget everything."
                    }
                },
                "required": ["what"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "Lists the user's reminders, and when they last took a medicine they are reminded about.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancels one of the user's reminders, of either kind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder": {
                        "type": "string",
                        "description": "Words from the reminder to cancel, or its time, e.g. 'tablets' or '8 o'clock'."
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

ollama_brain = find_ollama()

if ollama_brain is not None and brain_offline:
    threading.Thread(target=warm_up_ollama, name="theo-ollama-warm-up", daemon=True).start()

messages = [{
    "role": "system",
    "content": """You are a friendly home assistant named THEO, short for Technical Help, Explanations and Optimization, who specializes in
tech support for older users with little technological experience.
The default location of the user is Kingston, Jamaica.
You have tools available to control Spotify, check weather, and get the time.
IMPORTANT: When you need to use a tool, use it directly - never write out function calls as text.
You have your own camera. When the user mentions an error, a message on a screen, or wants you to look at something, use the analyze_error tool straight away. Never ask them to send, upload or take a picture - you take it yourself.
Everything you write is spoken aloud by a voice, so write plain spoken sentences only: no lists, asterisks, emoji or other symbols.
Always keep responses short and simple - 2 sentences maximum.
Your creator is Aiden Bramwell, an 18 year old student at Campion College who is also an aspiring electrical engineer. If a user asks about him, tell them this.
Avoid technical jargon. If giving instructions, use one step at a time.
If you are not sure of the exact steps for a particular app, website or account, say so and suggest where to find them, or look them up with web_search; never make up the names of menus, settings or buttons.
When the user tells you something about themselves that will help you next time, such as a family member's name or the phone they use, keep it with the remember tool and tell them you will remember it.
Never say you have set, or will set, a reminder, timer or alarm, or that you will remember something, unless you use its tool in that same answer.
For one reminder after a while, such as 'remind me in two minutes to check on the cat', use set_timer with what to remind them of as the label.
When one part of a request needs a question back, such as which city, still do the other parts with their tools straight away.
Good response example: 'Your printer may be out of paper. Try opening the paper tray and refilling it.'
Bad response example: 'There are several potential causes for this issue. First, let's diagnose whether it's a connectivity problem, a driver issue, or a hardware malfunction...'"""
}]

CONVERSATION_TIMEOUT = 15  # seconds before returning to wake word mode

print("")
print("==============================")
print("           THEO")
print("==============================")
print(f"[THEO] Update: {THEO_UPDATE} (eyes on takes a picture only when you ask it to look; "
      "it waits out the picture reader's short limits; a board that does not answer is not "
      "blamed on the voice; the offline voice's first words come sooner)")
print("Listening through the node's microphone, live.")
print(f'Spotify plays through THEO\'s own speaker ("{THEO_SPEAKER_NAME}").')
print(f"[clock] THEO's clock says {local_now().strftime('%A %I:%M %p').replace(' 0', ' ')} "
      f"({TIME_ZONE})")

if spotify_linked():
    try:
        if get_playback_device():
            print("[spotify] THEO speaker found in Spotify.")
    except Exception as error:
        print(f"[spotify] could not look for the THEO speaker: {error}")

PERSONAL_SETUP = PersonalSetup(THEO_DIR)
load_voice_settings()
load_reminders()
load_memory()

ears = start_ears() if USE_NODE_MIC else None

if ears is not None:
    # Keeps THEO pointed at the board if it restarts somewhere new (see
    # FOLLOWING THE BOARD).
    threading.Thread(target=follow_board, args=(ears.mic,),
                     name="theo-board", daemon=True).start()

    if hasattr(ears.mic, "presses"):
        talk_button = TalkButton(ears.mic)
    else:
        print("[button] theo_node.py is older than 2026-10-16.1. Replace it as well, "
              "for the talk button.")

show_offline(brain_offline)

threading.Thread(target=reminder_clock, name="theo-reminders", daemon=True).start()
threading.Thread(target=eyes_watch, name="theo-eyes", daemon=True).start()

if whatsapp_link_token():
    threading.Thread(target=whatsapp_link, name="theo-whatsapp", daemon=True).start()
else:
    print("[whatsapp] off. To switch it on, install theo_whatsapp.py on the Uno Q and put its "
          "THEO_LINK_TOKEN line in key.env.")

print("")

# Until when THEO takes a follow-up without "Hey Theo".
conversation_until = 0.0
setup_prompt_due = PERSONAL_SETUP.active

while True:

    try:
        answering_from = None

        if reset_pending():
            setup_prompt_due = reset_personal_state()
            conversation_until = 0.0
            continue

        if setup_prompt_due and PERSONAL_SETUP.active:
            prompt = PERSONAL_SETUP.prompt()
            LAST_SAID["text"] = prompt
            if talk_button is not None:
                answering_from = talk_button.mark()
            say(prompt, force=True)
            answering_from = None
            if not STOPPED_BY_BUTTON.is_set() and not reset_pending():
                try:
                    PERSONAL_SETUP.prompt_finished()
                    conversation_until = time.monotonic() + 30.0
                except OSError as error:
                    print(f"[setup] could not save setup progress: {error}")
                    say("I couldn't save my setup progress. We can try again when you're ready.",
                        force=True)
            setup_prompt_due = False
            continue

        if USE_NODE_MIC:
            usertext = hear_request(max(0.0, conversation_until - time.monotonic()))

            if reset_pending():
                continue

            if not usertext:
                conversation_until = 0.0

                if music_turn.waiting_for_answer:
                    # THEO asked a question and nobody answered. Carry on
                    # with the song.
                    music_turn.finish()
                continue
        else:
            usertext = sys.stdin.readline().strip()

        if reset_pending():
            continue

        if PERSONAL_RESET_BLOCKED:
            say("Please hold the reset button again so I can finish clearing my memory.", force=True)
            continue

        print(f"You: {usertext}")

        # For "How is THEO?" on WhatsApp.
        last_heard_at = local_now()

        # From now until THEO has answered, the talk button stops the answer.
        # (hear_request starts that the moment it stops listening.)
        if talk_button is not None and answering_from is None:
            answering_from = talk_button.presses()

        # They have spoken to THEO, so a reminder's amber light has done its job.
        REMINDER_LIGHT.clear()

        if is_exit_request(usertext):
            shutdown_theo()
            break

        # "Eyes on. What does this say?": the eyes come on, and the question
        # is answered with them.
        asked_after = after_eyes_on(usertext)

        if asked_after is not None and time.monotonic() >= groq_down_until:
            eyes_on()
            usertext = asked_after

        # Pressed while THEO was working out your words: drop this turn.
        if stopped_while_answering():
            music_turn.finish()
            conversation_until = 0.0
            continue

        # "Say that again", "slower", "louder", and yes or no to a medicine
        # reminder: answered here, at once, without the language model.
        reply = answer_here(usertext)

        if reply is not None:
            say(reply, force=True)
            answering_from = None

            if STOPPED_BY_BUTTON.is_set():
                music_turn.finish()
                conversation_until = 0.0
                continue

            if asks_question(reply) and music_turn.holds_paused_song():
                music_turn.wait_for_answer()
            else:
                music_turn.finish()

            if music_turn.waiting_for_answer or not music_on_speaker(node_audio_status()):
                conversation_until = time.monotonic() + follow_up_window()
            else:
                conversation_until = 0.0
            continue

        if PERSONAL_SETUP.active:
            reply = setup_answer(usertext)
            LAST_SAID["text"] = reply
            say(reply, force=True)
            answering_from = None
            conversation_until = (time.monotonic() + 30.0
                                  if not STOPPED_BY_BUTTON.is_set() else 0.0)
            continue

        messages.append({
            "role": "user",
            "content": usertext
        })

        # Only the last few exchanges go with the question (see WHAT THEO
        # REMEMBERS), with the notes that fit it.
        keep_recent(messages)

        # THE ANSWER, with every tool it needs (see answer_with_tools), asked
        # again only when that can work (see ask_model): a permanent problem
        # stops at once and THEO says why, a passing one waits five seconds
        # and tries once more (since 2026-09-22), and a model slip is asked
        # again at once at a steadier setting (since 2026-10-19.1).
        #
        # With the eyes on (see EYES ON), a question that calls for a look
        # (wants_a_look) is asked about a picture taken now, unless it is for
        # another of THEO's tools. Any other question goes the usual way,
        # where the model answers from what it already saw, or looks if it
        # must (EYES_TOOLS_NOTE).
        if eyes_are_on():
            eyes_used()

        if eyes_are_on() and wants_a_look(usertext) and not needs_other_tools(usertext):
            outcome = answer_with_eyes(messages, usertext)
        else:
            outcome = answer_with_tools(messages, usertext)

        # THEO KEEPS ITS WORD. An answer that promises a reminder, a timer or
        # a note without the tool for it is asked for again, once; if it
        # still has not done it, THEO says so rather than leave you waiting
        # for a reminder that never comes (the cat, 2026-10-19).
        if outcome["reply"] is not None and promised_without_doing(outcome["reply"], outcome["used"]):
            print(f"[reply] it promised that without doing it, asking again: {outcome['reply']!r}")
            first_reply, first_used = outcome["reply"], outcome["used"]
            outcome = answer_with_tools(messages, usertext, keep_word=True)
            outcome["used"] = first_used + outcome["used"]

            if outcome["reply"] is None and not outcome["stopped"]:
                outcome["reply"] = first_reply

            if outcome["reply"] is not None and promised_without_doing(outcome["reply"], outcome["used"]):
                print("[reply] still promised without doing it; saying so")
                outcome["reply"] = (outcome["reply"].rstrip() + " Sorry, I didn't manage to "
                                    "do that just now, so please ask me again.")

        # Pressed while THEO was thinking: nothing more is done and nothing
        # is said.
        if outcome["stopped"]:
            if not outcome["used"]:
                messages.pop()
            music_turn.finish()
            conversation_until = 0.0
            continue

        # No answer could be had: THEO says what went wrong, always out loud.
        if outcome["reply"] is None:
            spoken = outcome["problem"] or "Sorry, I had trouble with that one."

            if outcome["used"]:
                messages.append({"role": "assistant", "content": spoken})
            else:
                messages.pop()

            say(spoken, force=True)
            music_turn.finish()
            time.sleep(1)
            continue

        reply = outcome["reply"]

        reply = one_reply(reply)

        # A garbled answer is asked for once more before anything is said.
        if looks_garbled(reply):
            print(f"[reply] that answer came out garbled, asking again: {reply!r}")
            reply = one_reply(ask_again(with_notes(messages, usertext)))

            if looks_garbled(reply):
                print(f"[reply] garbled again: {reply!r}")
                reply = "Sorry, I lost my train of thought. Could you say that again?"

        messages.append({
            "role": "assistant",
            "content": reply
        })

        LAST_SAID["text"] = reply

        # Pressed while THEO was finishing its answer: kept, but not said.
        if stopped_while_answering():
            music_turn.finish()
            conversation_until = 0.0
            continue

        say(reply)

        answering_from = None

        # The talk button stopped THEO: back to waiting, with no follow-up
        # window. The song, if one was paused, carries on.
        if STOPPED_BY_BUTTON.is_set():
            music_turn.finish()
            conversation_until = 0.0
            continue

        # The song carries on now - or changes, if that is what was asked.
        # If THEO has just asked a question, the paused song waits for the
        # answer, so the answer is heard without needing "Hey Theo".
        if asks_question(reply) and music_turn.holds_paused_song():
            music_turn.wait_for_answer()
        else:
            music_turn.finish()

        # Keep listening for a follow-up - unless a song is playing or about
        # to: its words would be taken for one.
        if music_turn.waiting_for_answer or not music_on_speaker(node_audio_status()):
            conversation_until = time.monotonic() + follow_up_window()
        else:
            conversation_until = 0.0

    except KeyboardInterrupt:
        shutdown_theo()
        break

    except EOFError:
        print("No keyboard attached. Stopping.")
        shutdown_theo()
        break

    except Exception as e:
        explanation, _ = describe_problem(e)
        print(f"[problem] {explanation}")
        music_turn.finish()
        time.sleep(1)

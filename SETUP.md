# THEO Setup Guide

This guide sets up THEO update **2026-10-25.1**, with an Arduino UNO Q as the brain and an ESP32-S3 as the nerves. For what THEO does and how it works, see [README.md](README.md).

## Contents

1. [What you need](#1-what-you-need)
2. [Wire the ESP32-S3](#2-wire-the-esp32-s3)
3. [Upload the ESP32-S3 firmware](#3-upload-the-esp32-s3-firmware)
4. [Install THEO on the UNO Q](#4-install-theo-on-the-uno-q)
5. [Add your API keys](#5-add-your-api-keys)
6. [Offline answers with Ollama](#6-offline-answers-with-ollama)
7. [Spotify](#7-spotify)
8. [First start and checks](#8-first-start-and-checks)
9. [Training your own wake word](#9-training-your-own-wake-word)
10. [Settings you can change](#10-settings-you-can-change)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What you need

### Hardware

| Part | Notes |
|---|---|
| Arduino UNO Q, 4 GB | With a USB-C supply and cable rated 5 V 3 A |
| ESP32-S3 development board, N16R8 | 16 MB flash, 8 MB PSRAM |
| INMP441 I2S microphone | |
| OV2640 camera module | |
| Adafruit NeoPixel Ring 16 | With a 330 Ω resistor for its data line |
| 8002B amplifier module and a passive speaker | Plus a 2 kΩ resistor, a 10 nF capacitor and a volume potentiometer |
| Two normally open momentary push buttons | One to talk, one to reset memory |
| Breadboard and jumper wires | |
| Optional: LiPo cell and a Type-C 5 V 2 A boost module | To run the ESP32-S3 side from a battery |

### Software

* Arduino App Lab, for running THEO's Python app on the UNO Q
* The Arduino IDE with the **esp32** boards package by Espressif (THEO is built with version 3.3.12)

### Accounts and API keys

| Service | Used for |
|---|---|
| [Groq](https://console.groq.com/) | Online speech recognition, answers and reading screens |
| [ElevenLabs](https://elevenlabs.io/) | THEO's online voice |
| [OpenWeatherMap](https://openweathermap.org/api) | Weather |
| [Spotify for Developers](https://developer.spotify.com/dashboard) | Music. Playing songs also needs a Spotify **Premium** account. |

---

## 2. Wire the ESP32-S3

Disconnect power before changing any wiring.

| Component | Connection to the ESP32-S3 |
|---|---|
| **INMP441 microphone** | VDD to 3.3 V, GND to GND, SCK (BCLK) to **GPIO 1**, WS to **GPIO 2**, SD to **GPIO 42**, L/R to GND |
| **OV2640 camera** | 3.3 V and GND; SIOD **40**, SIOC **41**; D0 to D7 on **4, 5, 6, 7, 8, 9, 16, 17**; PCLK **21**, VSYNC **38**, HREF **39**; RESET to 3.3 V, PWDN to GND. This module has its own oscillator, so XCLK is left unconnected. |
| **Speaker** | **GPIO 12** through a 2 kΩ resistor, with a 10 nF capacitor to GND after it, then the volume potentiometer, the **8002B amplifier** and the passive speaker |
| **NeoPixel Ring 16** | Data In to **GPIO 11** through the 330 Ω resistor; 5 V and GND |
| **Talk button** | Between **GPIO 13** and GND |
| **Memory-reset button** | Between **GPIO 14** and GND |

Notes:

* Each button needs only its two connections. The firmware turns on the ESP32-S3's internal pull-ups.
* The memory-reset button is separate from the board's own EN/RST button. Restarting the board does not clear THEO's memory.
* Join the grounds of the ESP32-S3, the ring and the amplifier supply.
* Power the 8002B amplifier straight from the 5 V supply rather than the board's 5V pin, with a supply of 2 A or more and a 470 to 1000 µF capacitor across it. A supply that dips during loud answers can restart the board.
* Keep the camera's ground wire firmly connected. A loose camera ground produces pictures that will not open.

### Power

| Part | Supply |
|---|---|
| UNO Q | USB-C, 5 V 3 A |
| ESP32-S3 | USB, or a LiPo through a Type-C 5 V 2 A boost module |
| NeoPixel ring | 5 V from the ESP32-S3 board |
| 8002B amplifier | Straight from the 5 V supply, grounds joined |

---

## 3. Upload the ESP32-S3 firmware

1. Open `theo_esp32_node.ino` in the Arduino IDE. If asked, let it move the sketch into a folder named `theo_esp32_node`.
2. Select the board **ESP32S3 Dev Module**. The sketch only uses libraries that come with the esp32 boards package.
3. Leave **PSRAM** disabled for this firmware version. With PSRAM on, the first picture after a quiet spell can be an old one.
4. The sketch's Wi-Fi name and password (`WIFI_SSID` and `WIFI_PASSWORD`) are blank. Either fill them in on your own copy, and never commit them, or leave them blank and set up Wi-Fi from your phone in step 7.
5. Check these settings near the top of the sketch:

   ```cpp
   #define RING_PIN               11
   #define RING_RGBW               0
   #define TALK_BUTTON_PIN        13
   #define RESET_BUTTON_PIN       14
   #define RESET_HOLD_MS        5000UL
   ```

   The NeoPixel Ring 16 is sold as RGBW, but it reads three colour values per light, so `RING_RGBW` must stay `0`.
6. Click **Verify**, then **Upload**.
7. To set up Wi-Fi from a phone: when no known network answers, the board opens its own Wi-Fi network called **THEO-Setup**. Join it on your phone, open `http://192.168.4.1/setup`, enter your network name and password, and press **Save**. The board remembers networks entered this way, even after a power cut or a new upload.
8. Open the Serial Monitor at **115200** baud. The start-up output should include:

   ```text
   Board update: 2026-10-25.1 (white camera pulse + held memory reset)
   [button] talk button on GPIO 13
   [reset] memory reset on GPIO 14; hold five seconds
   [ring] 16 lights on GPIO 11
   ```

   The board also prints a camera check and a microphone check every time it starts.

---

## 4. Install THEO on the UNO Q

### 4.1 Get the code

```bash
git clone https://github.com/470Aiden/THEO.git
cd THEO
```

Put `main.py`, `theo_node.py`, `theo_onboarding.py` and `find_node.py` together in the folder that holds THEO's `main.py` in Arduino App Lab. The `key.env` file and the model files below go in the same folder.

### 4.2 Install the Python libraries

THEO runs on Python 3.13 in App Lab. Install these libraries into the Python environment THEO's app uses. The same list works as a `requirements.txt`:

```text
groq
openai
python-dotenv
requests
httpx
numpy
scipy
soundfile
faster-whisper
kokoro-onnx
elevenlabs
openwakeword==0.4.0
spotipy
ddgs
```

With pip:

```bash
pip install groq openai python-dotenv requests httpx numpy scipy soundfile faster-whisper kokoro-onnx elevenlabs "openwakeword==0.4.0" spotipy ddgs
```

Use openWakeWord **0.4.0**. On Python 3.13, newer versions need `tflite-runtime`, which has no build for the UNO Q.

### 4.3 Add the model files

Put these files next to `main.py`:

| File | Where it comes from |
|---|---|
| `hey_theo.onnx` | Your own "Hey THEO" wake-word model (see [section 9](#9-training-your-own-wake-word)) |
| `melspectrogram.onnx`, `embedding_model.onnx`, `silero_vad.onnx` | The [openWakeWord v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1) |
| `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin` | The [kokoro-onnx model-files-v1.0 release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0). The full-size `kokoro-v1.0.onnx` also works. Without these files, the offline voice stays off. |
| `alarm_sound.wav` | Optional. THEO plays a built-in chime without it. |

Faster-Whisper's `base` model downloads by itself the first time THEO starts, so the first start needs an internet connection.

### 4.4 Keep personal files out of Git

THEO saves memory, reminders and settings next to `main.py`. Add these lines to `.gitignore` so they are never committed:

```text
key.env
.cache
.spotify_token
theo_memory.json
reminders.json
theo_settings.json
theo_onboarding.json
theo_reset_pending.json
last_picture.jpg
theo_input.wav
```

---

## 5. Add your API keys

Create a file named `key.env` next to `main.py`:

```env
GROQ_API_KEY=your_groq_api_key
ELEVEN_LABS_KEY=your_elevenlabs_key
WEATHER_API_KEY=your_openweathermap_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Optional: only when Ollama runs on another computer
# OLLAMA_URL=http://<that computer's address>:11434
# Optional: a different Ollama model (llama3.2:1b is the default)
# OLLAMA_MODEL=llama3.2:1b
```

Never share `key.env` or upload it to GitHub.

---

## 6. Offline answers with Ollama

THEO uses Ollama when Groq cannot be reached. Ollama can run on the UNO Q itself or on another computer on the same network.

To run it on the UNO Q:

1. Install Ollama with its Linux install script from [ollama.com](https://ollama.com/download/linux).
2. Free space on the UNO Q's system partition. Stop Ollama and delete its two NVIDIA folders, which the UNO Q cannot use (about 2.2 GB):

   ```bash
   sudo systemctl stop ollama
   sudo rm -rf /usr/local/lib/ollama/cuda_v12 /usr/local/lib/ollama/cuda_v13
   ```

3. Keep the models on the larger `/home/arduino` partition:

   ```bash
   mkdir -p /home/arduino/ollama-models
   sudo systemctl edit ollama
   ```

   Add these lines in the editor:

   ```ini
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0"
   Environment="OLLAMA_MODELS=/home/arduino/ollama-models"
   User=arduino
   Group=arduino
   ```

4. Restart Ollama and download the model:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ollama pull llama3.2:1b
   ```

Inside App Lab, `localhost` is THEO's app container, not the UNO Q. `OLLAMA_HOST=0.0.0.0` lets THEO reach Ollama from inside App Lab. Start Ollama before THEO, because THEO looks for it once at start-up. Running Ollama's install script again restores the NVIDIA folders.

---

## 7. Spotify

THEO plays music through its own speaker. A bridge on the UNO Q runs librespot, which appears in Spotify as a speaker called **THEO**, and streams the music to the ESP32-S3.

1. In the [Spotify developer dashboard](https://developer.spotify.com/dashboard), create an app and add this Redirect URI:

   ```text
   http://127.0.0.1:8000/callback
   ```

   Put its client ID and client secret in `key.env`.

2. Sign in once on a computer with a web browser. Put `spotify_login.py` and a copy of `key.env` in one folder, then run:

   ```bash
   pip install spotipy python-dotenv
   python spotify_login.py
   ```

   Sign in and press **Agree**. This saves a file called `.spotify_token`. Copy it onto the UNO Q, next to `main.py`. THEO renews the sign-in by itself after that.

3. On the UNO Q, outside App Lab, put `theo_spotify_bridge.py` and `install_theo_spotify.sh` in one folder and run:

   ```bash
   bash install_theo_spotify.sh
   ```

   The installer sets up librespot and runs the bridge as the service `theo-spotify`, which starts with the UNO Q. Running it again updates the bridge. Edit the copy you install from, because the installer replaces the installed copy.

4. Watch the bridge with:

   ```bash
   journalctl -u theo-spotify -f
   ```

The speaker name `THEO_SPEAKER_NAME` in `main.py` must match `SPEAKER_NAME` in `theo_spotify_bridge.py`.

---

## 8. First start and checks

1. Power the ESP32-S3 first, then start THEO's app in Arduino App Lab. Both boards must be on the same Wi-Fi network.
2. To start THEO whenever the UNO Q is powered, open THEO's app in App Lab, use the menu beside its name and switch on **Run at startup**.
3. THEO finds the ESP32-S3 on the network by itself, waits for it if it is not ready and follows it if its address changes. The board also answers at `theo-esp32.local`.
4. Check the App Lab output for:

   ```text
   [THEO] Update: 2026-10-25.1
   ```

Then try each feature:

- [ ] Say **"Hey THEO, what time is it?"** The ring fills while THEO listens, a point of light travels while it thinks, and the light blooms while it speaks.
- [ ] Press the talk button while THEO is idle, then ask a question. Press it again while THEO is answering: the answer stops.
- [ ] Say **"Hey THEO, I'm getting an error on my computer."** THEO asks you to hold up the screen, the whole ring pulses soft white during the five-second countdown and capture, and THEO explains what it sees.
- [ ] Say **"Hey THEO, speak more slowly"**, then **"speak faster"**.
- [ ] Set a short timer and check the chime and the amber ring.
- [ ] Press the memory-reset button briefly. Nothing should be cleared.
- [ ] When you are ready to clear THEO's personal memory, hold the reset button for five seconds and release it. THEO introduces himself, mentions that he can speak more slowly or faster, asks your name and what you find difficult with technology, then reassures you.
- [ ] Restart THEO and ask **"What do you remember about me?"** Only your new answers should be there.
- [ ] Disconnect the internet. The ring turns blue, and THEO still tells the time and sets timers.

The reset clears remembered notes, the conversation and last answer, timers and reminders, voice-speed and loudness preferences, and the last photo and recording. It keeps Wi-Fi, `key.env`, the Spotify sign-in and the model files. THEO must be running and connected to the ESP32-S3 for a reset hold to count.

---

## 9. Training your own wake word

Use the free openWakeWord training notebook by the creator of openWakeWord:

https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb#scrollTo=qgaKWIY6WlJ1

On Colab's Python 3.13 the notebook needs these changes:

1. In the first cell, install:

   ```text
   pip install piper-phonemize-fix onnxruntime==1.22.1 pyarrow==20.0.0
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   %env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
   ```

2. Delete the `ai_edge_litert` and `onnx2tf` lines in Step 2.
3. A session restart loses the `%env` setting, so run everything again after one.

An error about `onnx_tf` after `hey_theo.onnx` is saved, and failing `.tflite` lines after the `.onnx` download, are harmless. Before switching to a new model, test it with openWakeWord 0.4.0 at THEO's threshold of 0.4, using recordings of your own voice.

---

## 10. Settings you can change

In `main.py`:

| Setting | Default | What it does |
|---|---|---|
| `WAKE_THRESHOLD` | 0.4 | How sure the wake-word model must be; higher wakes less often |
| `CONVERSATION_TIMEOUT` | 15 | Seconds THEO listens for a follow-up without the wake phrase |
| `FAMILY_NAMES` | empty | Names added to the speech recognition hint |
| `TIME_ZONE` | America/Jamaica | THEO's clock |
| `ONLINE_MODEL`, `VISION_MODEL` | `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` | The Groq models |
| `KOKORO_VOICE` | af_heart | The offline voice |

In `theo_esp32_node.ino`:

| Setting | Default | What it does |
|---|---|---|
| `RING_PIN`, `RING_RGBW` | 11, 0 | The ring's data pin and colour format |
| `RING_BRIGHTNESS` | 120 | Ring brightness, 0 to 255 |
| `TALK_BUTTON_PIN` | 13 | The talk button; `-1` means no button |
| `RESET_BUTTON_PIN`, `RESET_HOLD_MS` | 14, 5000 | The memory-reset button and how long to hold it, in milliseconds |
| `MIC_GAIN` | 4 | Microphone gain |
| `PIXEL_CLOCK_DIVIDER` | 16 | Camera clock margin; raise it if pictures come out corrupted |

---

## 11. Troubleshooting

| Problem | What to check |
|---|---|
| The ring stays dark | The start-up line `[ring] 16 lights on GPIO ...` must name the pin the ring is wired to (11) |
| The ring's colours are wrong | `RING_RGBW` must be `0` for this ring |
| The board restarts during long or loud answers | Power the amplifier straight from the 5 V supply, with the grounds joined and the capacitor in place |
| Pictures will not open | Check the camera's ground wire; raise `PIXEL_CLOCK_DIVIDER` if the start-up camera check reports many repeated bytes |
| THEO cannot answer offline | Ollama must listen on `0.0.0.0` and have the model pulled, and must be running before THEO starts |
| The output says the offline voice is off | Put `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin` next to `main.py` |
| THEO switched to the offline voice | ElevenLabs failed twice in a row, for example when the key's credit limit is reached. THEO prints ElevenLabs' message in the output and uses the online voice again after a restart. |
| openWakeWord will not install | Use version 0.4.0 on the UNO Q's Python 3.13 |
| THEO says Spotify is not linked | Create `.spotify_token` with `spotify_login.py` and copy it next to `main.py` |
| Services on the UNO Q fail oddly | Check free space with `df -h /`. A full system partition upsets App Lab. |

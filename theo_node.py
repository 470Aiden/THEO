"""
theo_node.py
============

THEO's ears and eyes, living on the ESP32-S3 node.

The node is a small HTTP server sitting on your network. This module is
the Python side of that conversation: it finds the node, asks it for a
picture or a recording, checks that what came back really IS a picture
or a recording, and hands it to THEO.

Everything here is deliberately careful. The node is a small board on
Wi-Fi -- it can be rebooting, asleep, or on a different address than it
was yesterday. When something goes wrong this module raises an error
carrying a plain sentence THEO can say out loud, instead of dropping a
traceback in front of an eighty year old.

    from theo_node import TheoNode, NodeUnavailable, MicStream

    node = TheoNode()

    node.record_audio(3000, path="question.wav")
    node.capture_image(path="screen.jpg")
    node.set_ring("thinking")
    node.set_offline(True)               # the ring's offline colour

    mic = MicStream("192.168.0.13")      # the microphone, live
    mic.start()
    piece = mic.read(timeout=1.0)        # 80 ms of sound, or None
    mic.presses                          # talk button presses so far

Nothing here needs a library THEO does not already use: requests and
numpy.

Version 2026-10-25.1: a deliberate memory-reset hold travels with the
button frames as a separate flag; reset_requests counts those requests.
"""

import collections
import os
import socket
import struct
import threading
import time

import numpy as np
import requests


# ------------------------------------------------------------------
# WHERE THE NODE LIVES
# ------------------------------------------------------------------
#
# The name is tried first because it survives the router handing out a
# different address. The raw address is the fallback, because .local
# name lookup is unreliable on some Windows and Linux setups.
#
# If your node ends up on a new address, change the second entry.

DEFAULT_HOSTS = (
    "http://theo-esp32.local",
    "http://192.168.0.18",
)


# ------------------------------------------------------------------
# WHAT THE NODE WILL AND WILL NOT DO
# ------------------------------------------------------------------
#
# These match the limits built into the sketch. The board CLAMPS a
# recording request into this range silently -- ask it for 30 seconds
# and you get 10 with no complaint -- so this module clamps out loud
# instead, and tells you it did.

MIN_RECORD_MS = 100
MAX_RECORD_MS = 10000


class NodeUnavailable(Exception):
    """
    The node could not be reached, or sent back something unusable.

    Carries two messages: the usual technical one for the log, and
    `spoken`, a sentence that is safe to read out to a person.
    """

    def __init__(self, message, spoken=None):
        super().__init__(message)

        self.spoken = spoken or (
            "I can't reach my camera and microphone right now."
        )


class TheoNode:
    """The ESP32 node, as an object THEO can talk to."""

    def __init__(
        self,
        hosts=None,
        connect_timeout=4.0,
        capture_timeout=20.0
    ):
        self.hosts = tuple(hosts) if hosts else DEFAULT_HOSTS
        self.connect_timeout = connect_timeout
        self.capture_timeout = capture_timeout

        # Which address answered last time. Saved so we do not pay the
        # name-lookup cost on every single request.
        self._working_host = None

        # The light ring. One helper thread sends the newest state, so
        # telling the ring something never makes THEO wait.
        self._ring_lock = threading.Condition()
        self._ring_wanted = None
        self._ring_sent = None
        self._offline_wanted = None
        self._offline_sent = None
        self._ring_thread = None
        self._ring_warned = False

    # --------------------------------------------------------------
    # FINDING THE NODE
    # --------------------------------------------------------------

    def _base_url(self, force_rescan=False):
        """
        Return an address that answered a moment ago.

        Tries the remembered one first, then works through the list.
        Raises NodeUnavailable if none of them answer.
        """

        if self._working_host and not force_rescan:
            return self._working_host

        problems = []

        for host in self.hosts:
            try:
                response = requests.get(
                    f"{host}/status",
                    timeout=self.connect_timeout
                )

                response.raise_for_status()

                self._working_host = host

                return host

            except Exception as error:
                problems.append(f"{host}: {error}")

        raise NodeUnavailable(
            "No address answered. Tried:\n  " + "\n  ".join(problems),
            "I can't find my camera and microphone on the network. "
            "Please check they have power."
        )

    def _request(self, path, params=None, timeout=None, retry=True):
        """
        Fetch one thing from the node.

        On failure it re-scans the address list once and tries again,
        because the usual cause is the board having taken a new address
        after a reboot.
        """

        try:
            base = self._base_url()

            response = requests.get(
                f"{base}{path}",
                params=params,
                timeout=timeout or self.connect_timeout
            )

            response.raise_for_status()

            return response

        except NodeUnavailable:
            # _base_url already worked out exactly what is wrong and
            # said so in words a person can act on. Do not flatten that
            # into a vaguer message on the way out.
            raise

        except Exception as error:
            if not retry:
                raise NodeUnavailable(
                    f"{path} failed: {error}",
                    "Something went wrong reaching my camera and "
                    "microphone."
                )

            # Forget the remembered address and try the whole list again.
            self._working_host = None

            return self._request(
                path,
                params=params,
                timeout=timeout,
                retry=False
            )

    # --------------------------------------------------------------
    # IS IT THERE, AND IS IT WELL?
    # --------------------------------------------------------------

    def status(self):
        """
        Ask the node how it is doing.

        Returns the node's own report, e.g.

            {
                "device": "THEO-ESP32",
                "camera":     {"ready": True, "resolution": "640x480"},
                "microphone": {"ready": True, "sample_rate": 16000,
                               "channels": 1}
            }
        """

        return self._request("/status").json()

    def is_ready(self):
        """
        True only if the node answers AND both parts report ready.

        Use this at startup so THEO knows what it can offer, rather
        than discovering mid-conversation that the camera is missing.
        """

        try:
            state = self.status()

            return bool(
                state.get("camera", {}).get("ready")
                and state.get("microphone", {}).get("ready")
            )

        except NodeUnavailable:
            return False

    # --------------------------------------------------------------
    # THE CAMERA
    # --------------------------------------------------------------

    def capture_image(self, path=None):
        """
        Take one picture and return it as bytes.

        If `path` is given the picture is also written there, and the
        path is what comes back instead of the bytes.

        The picture is checked before it is handed over. A JPEG that
        does not begin and end correctly is not passed on as if it
        were fine -- that is the failure that wasted a whole day of
        this build, and it is worth one line of checking to never
        repeat it.
        """

        response = self._request(
            "/camera/capture",
            timeout=self.capture_timeout
        )

        data = response.content

        self._check_jpeg(data)

        if path:
            with open(path, "wb") as handle:
                handle.write(data)

            return path

        return data

    @staticmethod
    def _check_jpeg(data):
        """Refuse a picture that is obviously incomplete."""

        if len(data) < 128:
            raise NodeUnavailable(
                f"Picture was only {len(data)} bytes.",
                "The picture didn't come through properly."
            )

        if data[:3] != b"\xff\xd8\xff":
            raise NodeUnavailable(
                "Data does not start like a JPEG "
                f"(first bytes {data[:3].hex(' ').upper()}).",
                "What came back from the camera wasn't a picture."
            )

        if data[-2:] != b"\xff\xd9":
            raise NodeUnavailable(
                "Picture is cut short -- the end marker is missing "
                f"(last bytes {data[-2:].hex(' ').upper()}).",
                "The picture was cut off on the way. Let me try again."
            )

    # --------------------------------------------------------------
    # THE MICROPHONE
    # --------------------------------------------------------------

    def record_audio(self, duration_ms=3000, path=None):
        """
        Record for a while and return the audio as bytes.

        If `path` is given the recording is also written there, and the
        path is what comes back instead of the bytes -- which is the
        form transcribe() wants.

        The board will not record for longer than ten seconds. Ask for
        more and it quietly gives you ten, so this clamps first and
        prints a note, rather than letting you believe you got what you
        asked for.
        """

        requested = int(duration_ms)

        clamped = max(MIN_RECORD_MS, min(MAX_RECORD_MS, requested))

        if clamped != requested:
            print(
                f"[node] Asked for {requested} ms; the board's limit is "
                f"{MIN_RECORD_MS}-{MAX_RECORD_MS} ms, so recording "
                f"{clamped} ms."
            )

        # The read has to outlast the recording itself, with room for
        # the board to assemble and send the file afterwards.
        timeout = (clamped / 1000.0) + 10.0

        response = self._request(
            "/audio/record",
            params={"duration": clamped},
            timeout=timeout
        )

        data = response.content

        self._check_wav(data)

        if path:
            with open(path, "wb") as handle:
                handle.write(data)

            return path

        return data

    @staticmethod
    def _check_wav(data):
        """
        Refuse a recording that is obviously incomplete.

        A WAV file states its own length in its header. If the bytes
        that arrived disagree with that number, the file was cut short
        in transit and Whisper will either fail or -- worse -- quietly
        transcribe half a sentence.
        """

        if len(data) < 44:
            raise NodeUnavailable(
                f"Recording was only {len(data)} bytes.",
                "The recording didn't come through properly."
            )

        if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise NodeUnavailable(
                "Data is not a WAV file "
                f"(first bytes {data[:4]!r}).",
                "What came back from the microphone wasn't audio."
            )

        declared = int.from_bytes(data[4:8], "little") + 8

        if declared != len(data):
            raise NodeUnavailable(
                f"Recording says it is {declared} bytes but "
                f"{len(data)} arrived -- it was cut short.",
                "The recording was cut off on the way. "
                "Let me try that again."
            )

    def stop_microphone(self):
        """
        Shut the microphone down on the board.

        Only needed if you want the camera working with the microphone
        completely out of the picture.
        """

        return self._request("/mic/stop").json()

    # --------------------------------------------------------------
    # THE LIGHT RING
    # --------------------------------------------------------------

    RING_STATES = ("idle", "listening", "thinking", "looking", "reminder", "off")

    def set_ring(self, state):
        """
        Tell the light ring what THEO is doing: "idle", "listening",
        "thinking", "looking", "reminder" (a slow amber pulse, after a
        reminder, until the person answers it) or "off". ("speaking" the
        board shows by itself, whenever THEO's voice plays.)

        Returns at once. A helper thread sends it, newest state first, so
        a slow network never holds up the conversation. The ring is only
        lights: if the board misses one, nothing else is affected.
        """

        if state not in self.RING_STATES:
            raise ValueError(f"unknown ring state: {state!r}")

        with self._ring_lock:
            self._ring_wanted = state
            self._start_ring_sender()

    def set_offline(self, offline):
        """
        True while THEO is running offline: the ring then shows its states
        in the offline colour, and a dim light at the top while it waits.
        Returns at once, like set_ring.
        """

        with self._ring_lock:
            self._offline_wanted = bool(offline)
            self._start_ring_sender()

    def _start_ring_sender(self):
        """Call with _ring_lock held."""

        if self._ring_thread is None:
            self._ring_thread = threading.Thread(
                target=self._ring_sender, name="theo-ring", daemon=True)
            self._ring_thread.start()

        self._ring_lock.notify()

    def _ring_sender(self):
        while True:
            with self._ring_lock:
                while (self._ring_wanted == self._ring_sent and
                       self._offline_wanted == self._offline_sent):
                    self._ring_lock.wait()

                state = self._ring_wanted
                offline = self._offline_wanted

            params = {}

            if state is not None:
                params["state"] = state

            if offline is not None:
                params["offline"] = "1" if offline else "0"

            base = self._working_host or self.hosts[0]

            try:
                response = requests.get(
                    f"{base}/ring",
                    params=params,
                    timeout=1.5
                )
                response.raise_for_status()

            except Exception as error:
                if not self._ring_warned:
                    print(f"[ring] the board did not take the ring state: {error}")
                    self._ring_warned = True

            with self._ring_lock:
                self._ring_sent = state
                self._offline_sent = offline


# ------------------------------------------------------------------
# THE MICROPHONE, LIVE
# ------------------------------------------------------------------
#
# For "Hey Theo". The board sends everything its microphone hears, as it
# hears it, over one connection to port 82: 16,000 samples a second,
# 16 bits each. MicStream keeps that connection open, cuts the sound into
# pieces and keeps the last 20 seconds of them for whoever reads.
#
# If the connection drops - the board restarting, Wi-Fi blinking - it
# reconnects by itself, and read() simply waits meanwhile.
#
# THE TALK BUTTON comes the same way (board 2026-10-16.1). Greeted with
# THEOMIC2 instead of THEOMIC1, the board answers THEOMIC2, the sample rate
# and its count of presses so far, then sends frames: one byte saying what
# the frame is, one spare byte, the frame's length (2 bytes), then the
# frame. "S" is sound, as before; "P" is a press of the talk button, with
# the board's count of presses (4 bytes) and whether it cut THEO's voice
# off (1 byte). An older board closes the connection at THEOMIC2; then the
# stream says THEOMIC1, and simply has no button.

MIC_PORT = 82
MIC_GREETING = b"THEOMIC1"
MIC_GREETING_BUTTON = b"THEOMIC2"
MIC_RATE = 16000


class GreetingRefused(ConnectionError):
    """The board closed the connection at the greeting: it does not speak that protocol."""


class MicStream:
    """The node's microphone, live, and the talk button's presses."""

    def __init__(self, host, port=MIC_PORT, piece=1280, keep_seconds=20.0):
        # A bare address or name: "192.168.0.13", not "http://...".
        self.host = host.split("://")[-1].split("/")[0]
        self.port = port

        # 1280 samples is 80 ms: the size the wake word model wants.
        self.piece = piece

        self._pieces = collections.deque(
            maxlen=max(1, int(keep_seconds * MIC_RATE / piece)))
        self._ready = threading.Condition()
        self._stopping = threading.Event()
        self._thread = None
        self._socket = None

        self.connected = False
        self.problem = None          # why it is not connected, for the log
        self.pieces_dropped = 0      # pieces nobody read in time

        # The talk button. presses counts every press since this started;
        # board_presses is the board's own count, which THEO's voice carries
        # back to it (see main.py's send_to_node).
        self.button = None           # True: the board sends presses; False: an older board
        self.presses = 0
        self.board_presses = 0
        self.last_press_cut_voice = False
        self.reset_requests = 0

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="theo-mic", daemon=True)
            self._thread.start()

        return self

    def stop(self):
        self._stopping.set()

        sock = self._socket

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        with self._ready:
            self._ready.notify_all()

    def read(self, timeout=None):
        """
        The next 80 ms of sound, oldest first, as int16 numbers - or None
        if none arrived within `timeout` seconds.
        """

        deadline = None if timeout is None else time.monotonic() + timeout

        with self._ready:
            while not self._pieces:
                if self._stopping.is_set():
                    return None

                remaining = None if deadline is None else deadline - time.monotonic()

                if remaining is not None and remaining <= 0:
                    return None

                self._ready.wait(remaining)

            return self._pieces.popleft()

    def drain(self):
        """Throw away everything heard so far. Returns how many pieces that was."""

        with self._ready:
            count = len(self._pieces)
            self._pieces.clear()

        return count

    def _run(self):
        delay = 1.0

        # Ask for the button first, every time a connection starts afresh, so
        # a board that has just been updated is noticed.
        greeting = MIC_GREETING_BUTTON

        while not self._stopping.is_set():
            try:
                self._listen_once(greeting)
                delay = 1.0

            except GreetingRefused as refused:
                if greeting == MIC_GREETING_BUTTON:
                    # An older board: say it the old way, straight away.
                    greeting = MIC_GREETING
                    self.button = False
                    self.connected = False
                    self._socket = None
                    continue

                self.problem = str(refused)
                greeting = MIC_GREETING_BUTTON

            except Exception as error:
                self.problem = str(error) or error.__class__.__name__

                # A connection that had started has ended: next time, ask
                # for the button again.
                if self.button is not None:
                    greeting = MIC_GREETING_BUTTON

            finally:
                self.connected = False
                self._socket = None

            if not self._stopping.is_set():
                # Try again soon, a little less often while it keeps failing.
                self._stopping.wait(delay)
                delay = min(delay * 2, 5.0)

    def _listen_once(self, greeting=MIC_GREETING):
        sock = socket.create_connection((self.host, self.port), timeout=4.0)
        self._socket = sock

        try:
            sock.settimeout(4.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(greeting)

            try:
                hello = self._receive_exactly(sock, 16)
            except ConnectionError as closed:
                raise GreetingRefused(
                    "port 82 closed the connection at the greeting - is the board "
                    "running the 2026-10-10.1 update?") from closed

            if hello[:8] != greeting:
                raise GreetingRefused(
                    "port 82 did not answer like THEO's board - is the "
                    "board running the 2026-10-10.1 update?")

            rate = struct.unpack("<I", hello[8:12])[0]

            if rate != MIC_RATE:
                raise ConnectionError(f"the board sends {rate} samples a second, not {MIC_RATE}")

            self.connected = True
            self.problem = None

            if greeting == MIC_GREETING_BUTTON:
                self.button = True
                self.board_presses = struct.unpack("<I", hello[12:16])[0]
                self._read_frames(sock)
            else:
                self._read_sound(sock)

        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _keep_sound(self, buffer):
        """Cut whole pieces off the front of `buffer` and keep them for read()."""

        want = self.piece * 2

        while len(buffer) >= want:
            piece = np.frombuffer(bytes(buffer[:want]), dtype="<i2").astype(np.int16)
            del buffer[:want]

            with self._ready:
                if len(self._pieces) == self._pieces.maxlen:
                    self.pieces_dropped += 1

                self._pieces.append(piece)
                self._ready.notify()

    def _read_sound(self, sock):
        """The THEOMIC1 stream: nothing but sound."""

        buffer = bytearray()

        while not self._stopping.is_set():
            # Nothing at all for 4 seconds means the board has gone.
            data = sock.recv(8192)

            if not data:
                raise ConnectionError("the board closed the microphone connection")

            buffer.extend(data)

            self._keep_sound(buffer)

    def _read_frames(self, sock):
        """The THEOMIC2 stream: sound, and the talk button's presses."""

        buffer = bytearray()

        while not self._stopping.is_set():
            # Nothing at all for 4 seconds means the board has gone.
            header = self._receive_exactly(sock, 4)
            kind = header[0:1]
            length = struct.unpack("<H", header[2:4])[0]
            body = self._receive_exactly(sock, length) if length else b""

            if kind == b"S":
                buffer.extend(body)
                self._keep_sound(buffer)

            elif kind == b"P":
                if len(body) >= 4:
                    self.board_presses = struct.unpack("<I", body[:4])[0]

                self.last_press_cut_voice = len(body) >= 5 and body[4] != 0
                if len(body) >= 6 and body[5] == 1:
                    self.reset_requests += 1
                self.presses += 1

                with self._ready:
                    self._ready.notify_all()

            else:
                raise ConnectionError(f"the board sent a frame THEO does not know ({kind!r})")

    @staticmethod
    def _receive_exactly(sock, count):
        data = bytearray()

        while len(data) < count:
            chunk = sock.recv(count - len(data))

            if not chunk:
                raise ConnectionError("the board closed the microphone connection")

            data.extend(chunk)

        return bytes(data)


# ------------------------------------------------------------------
# A QUICK CHECK YOU CAN RUN ON ITS OWN
# ------------------------------------------------------------------
#
#     python theo_node.py
#
# Proves the node is reachable and that both halves work, without
# involving THEO, an API key, or the internet.

if __name__ == "__main__":

    node = TheoNode()

    print("Looking for the node...")

    try:
        state = node.status()

    except NodeUnavailable as error:
        print("FAILED:", error)
        print("Spoken:", error.spoken)
        raise SystemExit(1)

    print("Found it at", node._working_host)
    print("Status:", state)

    print("\nTaking a picture...")
    picture = node.capture_image(path="node_test.jpg")
    print("Saved", picture, os.path.getsize(picture), "bytes")

    print("\nRecording three seconds...")
    recording = node.record_audio(3000, path="node_test.wav")
    print("Saved", recording, os.path.getsize(recording), "bytes")

    print("\nBoth halves work.")

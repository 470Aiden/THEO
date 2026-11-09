import json
import os
from pathlib import Path
import re
import tempfile


INTRODUCTION = (
    "Hello, I'm THEO, your Technical Help, Explanations and Optimization assistant. "
    "I can guide you through technology one step at a time, look at a confusing screen, "
    "play music, and help you keep reminders. "
    "You can ask me to speak more slowly or faster whenever you need. "
    "What would you like me to call you? You can say skip if you prefer."
)
NAME_QUESTION = "What would you like me to call you? You can say skip if you prefer."
CHALLENGE_QUESTION = (
    "What do you find difficult when using your phone or other technology? "
    "You can say skip if you'd rather tell me later."
)


class PersonalSetup:
    PERSONAL_FILES = (
        "theo_memory.json", "reminders.json", "theo_settings.json",
        "theo_onboarding.json", "last_picture.jpg", "theo_input.wav",
    )

    def __init__(self, directory):
        self.directory = Path(directory)
        self.file = self.directory / "theo_onboarding.json"
        self.marker = self.directory / "theo_reset_pending.json"
        self.state = {"stage": "complete", "name": ""}
        if self.marker.exists():
            with self.marker.open() as handle:
                marker = json.load(handle)
            if marker != {"reset": "personal-memory", "version": 1}:
                raise ValueError("THEO's pending reset record could not be verified")
            self.reset()
        if self.file.exists():
            with self.file.open() as handle:
                state = json.load(handle)
            if (not isinstance(state, dict)
                    or state.get("stage") not in ("intro", "name", "challenges", "complete")
                    or not isinstance(state.get("name", ""), str)):
                raise ValueError("THEO's onboarding record could not be read safely")
            self.state = state

    @property
    def active(self):
        return self.state["stage"] != "complete"

    def sync_directory(self):
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_json(self, path, value):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.directory,
                                             prefix=path.name + "-", suffix=".part",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=1)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self.sync_directory()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save_state(self, stage, name=""):
        state = {"stage": stage, "name": name}
        self.write_json(self.file, state)
        self.state = state

    def reset(self):
        self.write_json(self.marker, {"reset": "personal-memory", "version": 1})
        for name in self.PERSONAL_FILES:
            for suffix in ("", ".part"):
                (self.directory / (name + suffix)).unlink(missing_ok=True)
            for temporary in self.directory.glob(name + "-*.part"):
                temporary.unlink()
        self.save_state("intro")
        self.marker.unlink()
        self.sync_directory()

    def prompt(self):
        if self.state["stage"] == "intro":
            return INTRODUCTION
        if self.state["stage"] == "name":
            return NAME_QUESTION
        if self.state["stage"] == "challenges":
            return CHALLENGE_QUESTION
        return None

    def prompt_finished(self):
        if self.state["stage"] == "intro":
            self.save_state("name")

    @staticmethod
    def skipped(text):
        words = " ".join(re.findall(r"[a-z]+", text.casefold().replace("'", "")))
        return words in (
            "skip", "skip please", "no thanks", "rather not say", "id rather not say",
            "i dont want to say", "nothing", "nothing right now", "tell you later",
        )

    def answer(self, text, notes, added):
        text = " ".join(str(text or "").split()).strip()
        if not text:
            return self.prompt(), list(notes)
        if self.state["stage"] in ("intro", "name"):
            name = "" if self.skipped(text) else re.sub(
                r"^(?:my name is|my names|i am|i'm|call me|you can call me)\s+",
                "", text, flags=re.IGNORECASE).strip(" .!?")
            if name and (len(name) > 64 or len(name.split()) > 5
                         or not any(letter.isalpha() for letter in name)):
                return NAME_QUESTION, list(notes)
            self.save_state("challenges", name)
            greeting = f"It's nice to meet you, {name}. " if name else "That's okay. "
            return greeting + CHALLENGE_QUESTION, list(notes)
        if self.state["stage"] == "challenges":
            if len(text) > 400:
                return "Could you tell me one or two things you find difficult?", list(notes)
            kept = [dict(note) for note in notes]
            name = self.state.get("name", "")
            facts = []
            if name:
                facts.append((f"The user's name is {name}.", "family"))
            if not self.skipped(text):
                facts.append((f"The user finds these technical tasks difficult: {text}",
                              "phone and devices"))
            for fact, kind in facts:
                if not any(note.get("text", "").casefold() == fact.casefold() for note in kept):
                    kept.append({"text": fact, "kind": kind, "added": added})
            self.write_json(self.directory / "theo_memory.json", kept)
            self.save_state("complete")
            assurance = ("I can help you work through those difficulties, one step at a time. "
                         if not self.skipped(text)
                         else "I can help you with technology whenever you're ready. ")
            return (
                "Thank you. " + assurance +
                "If something is difficult, we can work through it together. "
                "You can say 'speak more slowly' or 'speak faster' whenever you need. "
                "Say 'Hey THEO' or press the talk button when you'd like help.",
                kept,
            )
        return None, list(notes)

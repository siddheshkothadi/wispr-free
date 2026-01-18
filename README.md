# Whispr (Local Voice Typing)

Local voice dictation for Windows. Hold a hotkey, speak, release — it types into whatever app you’re using. Built by Codex.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Use

- Hold `Ctrl+Windows` to record
- Release to transcribe + type

## Notes

- 100% local: Whisper via `faster-whisper`
- Requires: Windows, Python 3.8+, microphone

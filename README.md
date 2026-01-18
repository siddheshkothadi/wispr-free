# Wispr (Local Voice Typing)

A free, private, and local voice dictation tool inspired by **Wispr Flow**, vibe coded entirely using the **Gemini CLI**.

## What is it?

Wispr is a lightweight background application for Windows that allows you to type with your voice in **any** application. It runs 100% locally on your machine, meaning your voice data never leaves your computer.

## Features

*   **Global Hotkey:** Hold `Ctrl + Win` to record.
*   **Universal Typing:** Automatically types into the active text field (Notepad, Slack, VS Code, Browser, etc.).
*   **100% Local & Private:** Uses OpenAI's **Whisper** model running locally on your CPU/GPU. No API keys, no cloud costs.
*   **Modern UI:** Features a sleek, minimal "floating pill" interface with reactive wave animations.
*   **High Accuracy:** Powered by the `faster-whisper` library for industry-leading speech recognition.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:**
    ```bash
    python main.py
    ```

3.  **Use It:**
    *   Go to any text box.
    *   **Hold** `Ctrl + Windows`.
    *   Speak your sentence.
    *   **Release** keys to transcribe and type.

## Requirements

*   Python 3.8+
*   Windows (for the overlay and hotkeys)
*   A microphone

## Credits

*   Built with **Gemini CLI**.
*   Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
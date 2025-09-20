# ATC Live Transcriber and Explainer

This Python script listens to live Air Traffic Control (ATC) communications, transcribes them using OpenAI's Whisper, and sends the transcriptions to Cerebras AI for plain-language explanations that non-technical listeners can understand.

## Features

- **Real-time Audio Capture**: Captures audio from your microphone in 5-second chunks
- **Speech-to-Text Transcription**: Uses OpenAI Whisper for accurate ATC communication transcription
- **AI-Powered Explanations**: Leverages Cerebras AI to explain complex ATC communications in simple terms
- **Live Processing**: Processes audio continuously while you listen to ATC feeds

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: On macOS, you may need to install PortAudio first:

```bash
brew install portaudio
```

### 2. Configure API Keys

1. Copy the example environment file:

   ```bash
   cp env_example.txt .env
   ```

2. Edit `.env` and add your Cerebras API key:

   ```
   CEREBRAS_API_KEY=your_actual_api_key_here
   ```

3. Get your Cerebras API key from: https://cerebras.ai/

### 3. Audio Setup

Make sure your microphone is working and set as the default input device. The script will capture audio from your default microphone.

If you want to capture ATC audio from your speakers (e.g., from LiveATC.net), you may need to:

- Use a virtual audio cable to route system audio to the microphone input
- Or play the ATC feed through speakers while the script captures via microphone

## Usage

Run the script:

```bash
python atc_transcriber.py
```

The script will:

1. Load the Whisper model (this may take a moment on first run)
2. Start capturing audio from your microphone
3. Process audio in 5-second chunks
4. Transcribe any detected speech
5. Send transcriptions to Cerebras for explanation
6. Display both the raw transcription and plain-language explanation

Press `Ctrl+C` to stop the script.

## Example Output

```
🎤 Started recording audio...
Press Ctrl+C to stop

🎵 Processing audio chunk...

📝 Transcription: United 1234 contact departure 121.9

💡 Explanation: The air traffic controller is instructing United Airlines flight 1234 to switch radio frequencies and contact the departure control on frequency 121.9 MHz. This typically happens after takeoff when the aircraft is transitioning from tower control to departure control for further routing instructions.

--------------------------------------------------------------------------------
```

## Configuration

You can adjust audio settings by modifying the constants in `atc_transcriber.py`:

- `CHUNK`: Audio buffer size (default: 1024)
- `RATE`: Sample rate in Hz (default: 16000)
- `RECORD_SECONDS`: Length of audio chunks to process (default: 5)

## Troubleshooting

### Audio Issues

- Ensure your microphone is connected and set as the default input device
- Check audio levels - the script needs clear audio to transcribe effectively
- On macOS, you may need to grant microphone permissions to your terminal

### API Issues

- Verify your Cerebras API key is correct and has sufficient credits
- Check your internet connection
- The script will continue transcribing even if the Cerebras API is unavailable

### Performance

- The Whisper model runs locally and may be slow on older computers
- Consider using a smaller Whisper model (`tiny` or `small`) for faster processing
- Close other applications to free up system resources

## Technical Details

- **Audio Format**: 16-bit PCM, 16kHz sample rate, mono
- **Transcription**: OpenAI Whisper "base" model
- **AI Model**: Cerebras Llama 3.1 70B
- **Processing**: Multi-threaded with audio capture and processing running concurrently

## License

This project is provided as-is for educational and personal use.

#!/usr/bin/env python3
"""
ATC Live Transcriber and Explainer

This script captures live audio, transcribes ATC communications using Whisper,
and sends the transcription to Cerebras for plain-language explanation.
"""

import pyaudio
import wave
import threading
import queue
import time
import os
import tempfile
import whisper
import requests
import json
import signal
from contextlib import contextmanager
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()


@contextmanager
def timeout_context(seconds):
    """Context manager for timing out operations"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Clean up
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class ATCTranscriber:
    def __init__(self):
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3  # Process audio in 3-second chunks (shorter = faster)

        # Initialize components
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.chunks_processed = 0  # Track progress
        self.conversation_history = []  # Store recent transcriptions for context
        self.max_history_items = 5  # Keep last 5 transcriptions for context

        # Display microphone information
        self.display_microphone_info()

        # Load Whisper model (using 'tiny' for speed, 'base' for accuracy)
        # Model comparison:
        # - tiny: 39M params, ~32x faster than base, good for real-time
        # - base: 74M params, better accuracy but slower
        # - small: 244M params, even better accuracy, much slower
        # Using 'base' model for better accuracy (slower but more accurate than 'tiny')
        model_size = os.getenv("WHISPER_MODEL", "tiny")  # Can be set via .env file
        print(f"Loading Whisper '{model_size}' model...")
        self.whisper_model = whisper.load_model(model_size)
        print(f"Whisper '{model_size}' model loaded successfully!")

        # Performance optimization settings
        self.whisper_options = {
            "fp16": False,  # Use FP32 for CPU (FP16 is for GPU)
            "language": "en",  # Force English language (prevents wrong language detection)
            "task": "transcribe",  # Only transcribe, don't translate
            "no_speech_threshold": 0.6,  # Skip silent audio faster
            "condition_on_previous_text": False,  # Disable context for speed
            "initial_prompt": "Air traffic control communication with aircraft callsigns, frequencies, and aviation terminology.",  # Bias towards ATC content
        }
        print(f"🔍 DEBUG: Whisper options configured: {self.whisper_options}")

        # Cerebras API settings
        self.cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
        self.cerebras_base_url = "https://api.cerebras.ai/v1"

        if not self.cerebras_api_key:
            print("Warning: CEREBRAS_API_KEY not found in environment variables")
            print("Please create a .env file with your Cerebras API key")

    def display_microphone_info(self):
        """Display information about the microphone being used"""
        try:
            # Get default input device info
            default_input_device = self.audio.get_default_input_device_info()

            print(f"🎙️  Microphone: {default_input_device['name']}")
            print(f"   • Device Index: {default_input_device['index']}")
            print(
                f"   • Max Input Channels: {default_input_device['maxInputChannels']}"
            )
            print(
                f"   • Default Sample Rate: {default_input_device['defaultSampleRate']:.0f} Hz"
            )

            # Check if device supports our desired format
            try:
                is_supported = self.audio.is_format_supported(
                    rate=self.RATE,
                    input_device=default_input_device["index"],
                    input_channels=self.CHANNELS,
                    input_format=self.FORMAT,
                )
                if is_supported:
                    print(f"   • Format Support: ✅ 16kHz mono supported")
                else:
                    print(f"   • Format Support: ⚠️  16kHz mono may not be optimal")
            except:
                print(f"   • Format Support: ❓ Unable to verify")

            print()

        except Exception as e:
            print(f"⚠️  Could not detect microphone info: {e}")
            print("🎙️  Using system default microphone")
            print()

    def list_all_audio_devices(self):
        """List all available audio input devices (for debugging)"""
        print("📋 Available Audio Input Devices:")
        print("-" * 50)

        device_count = self.audio.get_device_count()
        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:  # Only show input devices
                    marker = (
                        "🎙️  "
                        if i == self.audio.get_default_input_device_info()["index"]
                        else "   "
                    )
                    print(f"{marker}[{i}] {device_info['name']}")
                    print(
                        f"      Channels: {device_info['maxInputChannels']}, Rate: {device_info['defaultSampleRate']:.0f} Hz"
                    )
            except:
                continue
        print()

    def start_recording(self):
        """Start recording audio in a separate thread"""
        self.is_recording = True

        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        print("🎤 Started recording audio...")
        print("Press Ctrl+C to stop")

        try:
            while self.is_recording:
                frames = []
                for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                    if not self.is_recording:
                        break
                    data = stream.read(self.CHUNK)
                    frames.append(data)

                if frames:
                    # Put audio data in queue for processing
                    audio_data = b"".join(frames)
                    self.audio_queue.put(audio_data)

        except KeyboardInterrupt:
            print("\n🛑 Stopping recording...")
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False

    def save_audio_chunk(self, audio_data: bytes) -> str:
        """Save audio data to a temporary WAV file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

        with wave.open(temp_file.name, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)

        return temp_file.name

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio using Whisper with performance optimizations"""
        try:
            # Start timing for performance monitoring
            start_time = time.time()

            # Use optimized settings for faster transcription
            # Note: Temporarily removed timeout for debugging
            print("🔍 DEBUG: About to call whisper.transcribe()...")
            print(f"🔍 DEBUG: Using optimized options: {self.whisper_options}")
            result = self.whisper_model.transcribe(
                audio_file_path, **self.whisper_options
            )
            print("🔍 DEBUG: whisper.transcribe() returned")

            transcription = result["text"].strip()

            # Debug: Show detected language and confidence
            detected_language = result.get("language", "unknown")
            print(f"🔍 DEBUG: Detected language: {detected_language}")
            if "language_probs" in result:
                top_languages = sorted(
                    result["language_probs"].items(), key=lambda x: x[1], reverse=True
                )[:3]
                print(f"🔍 DEBUG: Top language probabilities: {top_languages}")

            # Validate transcription quality
            if detected_language != "en":
                print(
                    f"⚠️  Warning: Detected language '{detected_language}' instead of English"
                )
                print("🔍 This might indicate poor audio quality or background noise")

            # Check for obvious non-English characters or patterns
            if self._contains_non_english_patterns(transcription):
                print(
                    f"⚠️  Warning: Transcription contains non-English patterns: '{transcription}'"
                )
                print("🔍 Skipping this chunk due to likely transcription error")
                os.unlink(audio_file_path)
                return None

            # Show processing time
            processing_time = time.time() - start_time
            print(f"⚡ Transcription completed in {processing_time:.1f}s")

            # Clean up temporary file
            os.unlink(audio_file_path)

            return transcription if transcription else None

        except TimeoutError as e:
            print(f"⏰ Transcription timeout: {e}")
            # Clean up temporary file on timeout
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            return None
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            # Clean up temporary file on error
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            return None

    def _contains_non_english_patterns(self, text: str) -> bool:
        """Check if transcription contains obvious non-English patterns"""
        if not text:
            return False

        # Check for non-ASCII characters (except common punctuation)
        import re

        # Allow basic ASCII, common punctuation, and numbers
        if re.search(r"[^\x00-\x7F]", text):
            return True

        # Check for patterns that suggest wrong language detection
        suspicious_patterns = [
            # Common non-English words that Whisper sometimes outputs
            r"\b(und|der|die|das|les|des|une|la|le|du|de|el|la|los|las)\b",
            # Repeated single characters (often transcription errors)
            r"\b[a-z]\s+[a-z]\s+[a-z]\b",
            # Very long words without spaces (concatenation errors)
            r"\b\w{25,}\b",
            # Multiple consecutive non-word characters
            r"[^\w\s]{3,}",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text.lower()):
                return True

        return False

    def explain_atc_communication(self, transcription: str) -> Optional[str]:
        """Send transcription to Cerebras for explanation with conversation context"""
        if not self.cerebras_api_key:
            return "⚠️  Cerebras API key not configured"

        try:
            headers = {
                "Authorization": f"Bearer {self.cerebras_api_key}",
                "Content-Type": "application/json",
            }

            # Build context from recent conversation history
            context_section = ""
            if self.conversation_history:
                context_section = "\n\nRecent ATC conversation context:\n"
                for i, prev_msg in enumerate(
                    self.conversation_history[-3:], 1
                ):  # Last 3 messages
                    context_section += f'{i}. "{prev_msg}"\n'
                context_section += "\n"

            prompt = f"""You are an expert air traffic control (ATC) interpreter analyzing TRANSCRIBED AUDIO.

IMPORTANT: The text you receive is from automatic speech recognition (ASR) and may contain:
- Misheard words or numbers (especially callsigns, frequencies, altitudes)
- Missing words or incomplete sentences
- Phonetic spelling errors (e.g., "tree" for "three", "niner" for "nine")
- Radio static effects on transcription accuracy

Please explain the following ATC communication in simple, plain language that a non-technical person can understand, while being aware of potential transcription errors.

Focus on:
- What is likely happening (takeoff, landing, routing, etc.) - consider context
- Who is probably involved (aircraft callsign, controller) - note if callsign seems unclear
- Any important safety or operational information
- The significance of any instructions given
- If something seems unclear due to transcription, mention it{context_section}
Current ATC Communication: "{transcription}"

Explanation:"""

            data = {
                "model": "llama-3.3-70b",  # Using Cerebras' Llama 3.3 70B model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 350,  # Increased for more detailed explanations with context
                "temperature": 0.7,
            }

            response = requests.post(
                f"{self.cerebras_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"❌ Cerebras API error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"⚠️  API Error: {response.status_code}"

        except Exception as e:
            print(f"❌ Cerebras API error: {e}")
            return f"⚠️  API Error: {str(e)}"

    def process_audio_queue(self):
        """Process audio chunks from the queue"""
        print("🔄 Started audio processing thread...")

        while self.is_recording or not self.audio_queue.empty():
            try:
                print("🔍 DEBUG: Waiting for audio data from queue...")
                # Get audio data from queue (with timeout)
                audio_data = self.audio_queue.get(timeout=1)
                print("🔍 DEBUG: Got audio data from queue")

                # Update progress tracking
                self.chunks_processed += 1
                queue_size = self.audio_queue.qsize()
                print(
                    f"🎵 Processing audio chunk #{self.chunks_processed}... ({queue_size} remaining in queue)"
                )

                print("🔍 DEBUG: Saving audio chunk to file...")
                # Save audio to temporary file
                audio_file = self.save_audio_chunk(audio_data)
                print(f"🔍 DEBUG: Saved audio to {audio_file}")

                print("🔍 DEBUG: Starting transcription...")
                # Transcribe audio
                transcription = self.transcribe_audio(audio_file)
                print(f"🔍 DEBUG: Transcription completed: '{transcription}'")

                if (
                    transcription and len(transcription.strip()) > 10
                ):  # Only process substantial transcriptions
                    print(f"\n📝 Transcription: {transcription}")

                    # Get explanation from Cerebras
                    explanation = self.explain_atc_communication(transcription)

                    if explanation:
                        print(f"💡 Explanation: {explanation}")

                    # Add to conversation history for context
                    self.conversation_history.append(transcription)
                    # Keep only the most recent items
                    if len(self.conversation_history) > self.max_history_items:
                        self.conversation_history.pop(0)

                    print("-" * 80)
                else:
                    print("🔇 No clear speech detected in audio chunk")

                print("🔍 DEBUG: Marking task as done...")
                # Mark task as done
                self.audio_queue.task_done()
                print("🔍 DEBUG: Task marked as done")

            except queue.Empty:
                print("🔍 DEBUG: Queue empty, continuing...")
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                # Still mark task as done even on error to prevent hanging
                try:
                    self.audio_queue.task_done()
                except ValueError:
                    pass  # task_done() called more times than items in queue

    def run(self):
        """Main method to run the ATC transcriber"""
        print("🚀 Starting ATC Live Transcriber...")
        print(
            "This will capture audio, transcribe ATC communications, and explain them."
        )
        print()

        # Start audio processing thread (not daemon so it can finish processing)
        processing_thread = threading.Thread(
            target=self.process_audio_queue, daemon=False
        )
        processing_thread.start()

        # Store reference to thread for proper shutdown
        self.processing_thread = processing_thread

        # Start recording (this will block until Ctrl+C)
        try:
            self.start_recording()
        except KeyboardInterrupt:
            pass

        print("⏳ Finishing processing remaining audio...")

        # Show initial queue size
        initial_queue_size = self.audio_queue.qsize()
        if initial_queue_size > 0:
            print(f"📊 {initial_queue_size} audio chunks remaining to process")

        # Wait for queue to be processed with timeout and progress updates
        timeout_seconds = 60  # Maximum wait time
        start_time = time.time()

        while (
            not self.audio_queue.empty()
            and (time.time() - start_time) < timeout_seconds
        ):
            current_size = self.audio_queue.qsize()
            if current_size > 0:
                print(f"⏳ Still processing... {current_size} chunks remaining")
            time.sleep(2)  # Check every 2 seconds

        # Final check
        if self.audio_queue.empty():
            print("✅ All audio chunks processed successfully")
        else:
            remaining = self.audio_queue.qsize()
            print(
                f"⚠️  Timeout reached. {remaining} chunks may not have been processed."
            )
            print("This can happen if processing is very slow or if there were errors.")

        # Wait for processing thread to finish (with timeout)
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            print("⏳ Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=10)
            if self.processing_thread.is_alive():
                print("⚠️  Processing thread did not finish within timeout")
            else:
                print("✅ Processing thread finished")

        print("✅ ATC Transcriber stopped")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "audio"):
            self.audio.terminate()


def main():
    """Main entry point"""
    import sys

    print("ATC Live Transcriber and Explainer")
    print("=" * 50)

    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-devices" or sys.argv[1] == "-l":
            print("Listing available audio devices...\n")
            transcriber = ATCTranscriber()
            transcriber.list_all_audio_devices()
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python atc_transcriber.py           # Run the transcriber")
            print("  python atc_transcriber.py -l        # List audio devices")
            print("  python atc_transcriber.py --help    # Show this help")
            return

    # Check for required environment variables
    if not os.getenv("CEREBRAS_API_KEY"):
        print("\n⚠️  Setup Required:")
        print("1. Create a .env file in this directory")
        print("2. Add your Cerebras API key: CEREBRAS_API_KEY=your_key_here")
        print("3. Get your API key from: https://cerebras.ai/")
        print()

        # Ask if user wants to continue without API key
        response = input("Continue without Cerebras integration? (y/N): ").lower()
        if response != "y":
            return

    transcriber = ATCTranscriber()
    transcriber.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fast ATC Live Transcriber and Explainer using ATC-fine-tuned Whisper

This script uses faster-whisper with ATC-specific fine-tuned models for better
accuracy on air traffic control communications.
"""

import pyaudio
import wave
import threading
import queue
import time
import os
import tempfile
import requests
import json
import signal
from contextlib import contextmanager
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster-whisper not installed!")
    print("Please install it with: pip install faster-whisper")
    exit(1)


class FastATCTranscriber:
    def __init__(self):
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3  # Process audio in 3-second chunks

        # Initialize components
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.chunks_processed = 0  # Track progress
        self.conversation_history = []  # Store recent transcriptions for context
        self.max_history_items = 5  # Keep last 5 transcriptions for context

        # Display microphone information
        self.display_microphone_info()

        # Load ATC-fine-tuned Whisper model
        model_name = os.getenv(
            "WHISPER_MODEL",
            "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper",
        )
        print(f"Loading ATC-fine-tuned Whisper model: {model_name}")
        print("This may take a moment on first run (downloading model)...")

        try:
            # Use faster-whisper with ATC-fine-tuned model
            self.whisper_model = WhisperModel(
                model_name,
                device="cpu",  # Use CPU on Mac (faster-whisper is optimized for this)
                compute_type="float32",  # Best quality for CPU
            )
            print(f"✅ ATC-fine-tuned Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load ATC model: {e}")
            print("Falling back to standard medium.en model...")
            self.whisper_model = WhisperModel(
                "medium.en", device="cpu", compute_type="float32"
            )

        # Transcription settings optimized for ATC
        self.transcribe_options = {
            "language": "en",
            "vad_filter": True,  # Voice Activity Detection for better ATC transmission detection
            "beam_size": 5,  # Better accuracy
            "best_of": 5,  # Multiple passes for best result
            "temperature": 0.0,  # Deterministic output
            "initial_prompt": "Air traffic control communication with aircraft callsigns, frequencies, altitudes, and aviation terminology.",
        }
        print(f"🔍 DEBUG: Transcription options: {self.transcribe_options}")

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
        """Transcribe audio using ATC-fine-tuned faster-whisper"""
        try:
            # Start timing for performance monitoring
            start_time = time.time()

            print("🔍 DEBUG: Starting ATC-fine-tuned transcription...")

            # Use faster-whisper with ATC-optimized settings
            segments, info = self.whisper_model.transcribe(
                audio_file_path, **self.transcribe_options
            )

            # Combine all segments into one transcription
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text.strip())

            transcription = " ".join(transcription_parts).strip()

            # Show processing time and model info
            processing_time = time.time() - start_time
            print(f"⚡ ATC transcription completed in {processing_time:.1f}s")
            print(
                f"🔍 DEBUG: Language: {info.language}, Avg log prob: {info.avg_logprob:.3f}"
            )

            # Clean up temporary file
            os.unlink(audio_file_path)

            return transcription if transcription else None

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            # Clean up temporary file on error
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            return None

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

            prompt = f"""You are an expert air traffic control (ATC) interpreter analyzing TRANSCRIBED AUDIO from an ATC-FINE-TUNED AI MODEL.

IMPORTANT: This transcription comes from a specialized ATC-trained Whisper model, so it should be more accurate for aviation terminology, but may still contain:
- Misheard callsigns or frequencies due to radio static
- Phonetic number variations (e.g., "niner" for "nine", "tree" for "three")
- Missing words during radio transmission gaps
- Overlapping transmissions

Please explain the following ATC communication in simple, plain language that a non-technical person can understand.

Focus on:
- What is likely happening (takeoff, landing, routing, etc.) - consider context
- Who is probably involved (aircraft callsign, controller) - note if callsign seems unclear
- Any important safety or operational information
- The significance of any instructions given
- If something seems unclear due to radio conditions, mention it{context_section}
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

                print("🔍 DEBUG: Starting ATC transcription...")
                # Transcribe audio using ATC-fine-tuned model
                transcription = self.transcribe_audio(audio_file)
                print(f"🔍 DEBUG: Transcription completed: '{transcription}'")

                if (
                    transcription
                    and len(transcription.strip()) > 5  # Lower threshold for ATC
                ):  # Process shorter ATC transmissions
                    print(f"\n📝 ATC Transcription: {transcription}")

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
                    print("📻 No clear ATC transmission detected in audio chunk")

                print("🔍 DEBUG: Marking task as done...")
                # Mark task as done
                self.audio_queue.task_done()
                print("🔍 DEBUG: Task marked as done")

            except queue.Empty:
                print("🔍 DEBUG: Queue empty, continuing...")
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                print(f"🔍 DEBUG: Exception details: {type(e).__name__}: {e}")
                # Still mark task as done even on error to prevent hanging
                try:
                    self.audio_queue.task_done()
                    print("🔍 DEBUG: Task marked as done after error")
                except ValueError:
                    print("🔍 DEBUG: task_done() called more times than items in queue")
                    pass

        print("🔍 DEBUG: Exited processing loop")

    def run(self):
        """Main method to run the ATC transcriber"""
        print("🚀 Starting Fast ATC Live Transcriber...")
        print("Using ATC-fine-tuned Whisper model for better aviation accuracy!")
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

        print("✅ Fast ATC Transcriber stopped")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "audio"):
            self.audio.terminate()


def main():
    """Main entry point"""
    import sys

    print("Fast ATC Live Transcriber and Explainer")
    print("Using ATC-Fine-Tuned Whisper Models")
    print("=" * 50)

    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-devices" or sys.argv[1] == "-l":
            print("Listing available audio devices...\n")
            transcriber = FastATCTranscriber()
            transcriber.list_all_audio_devices()
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python fastatc_transcriber.py           # Run the transcriber")
            print("  python fastatc_transcriber.py -l        # List audio devices")
            print("  python fastatc_transcriber.py --help    # Show this help")
            print("\nEnvironment Variables:")
            print("  WHISPER_MODEL=model_name    # Override ATC model")
            print("  CEREBRAS_API_KEY=your_key   # For explanations")
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

    transcriber = FastATCTranscriber()
    transcriber.run()


if __name__ == "__main__":
    main()

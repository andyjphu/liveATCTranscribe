#!/bin/bash

echo "🚀 Setting up Fast ATC Transcriber with ATC-Fine-Tuned Models"
echo "============================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup script is designed for macOS"
    echo "You may need to adjust commands for your system"
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3 and try again"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Fast ATC Transcriber requirements..."
pip install -r requirements-fast.txt

# Check if PortAudio is needed (for pyaudio)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "⚠️  Homebrew not found. You may need to install PortAudio manually if pyaudio fails"
    else
        echo "🍺 Installing PortAudio via Homebrew (for pyaudio)..."
        brew install portaudio
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Copy env_example.txt to .env and add your Cerebras API key"
echo "3. Run the transcriber: python fastatc_transcriber.py"
echo ""
echo "🎙️  The first run will download the ATC-fine-tuned model (~500MB)"
echo "📊 This may take a few minutes but will be cached for future use"
echo ""
echo "🔧 Available commands:"
echo "   python fastatc_transcriber.py         # Run transcriber"
echo "   python fastatc_transcriber.py -l      # List audio devices"
echo "   python fastatc_transcriber.py --help  # Show help"

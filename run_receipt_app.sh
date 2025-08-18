#!/bin/bash

# Receipt OCR Handler - Streamlit App Launcher
# This script sets up the environment and launches the Streamlit app

echo "🧾 Starting Receipt OCR Handler..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
if ! python -c "import streamlit, langchain, sqlalchemy" 2>/dev/null; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

# Check if PostgreSQL is running (macOS)
if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo "⚠️  Warning: PostgreSQL doesn't seem to be running."
    echo "   Please start PostgreSQL before running the app."
    echo "   On macOS: brew services start postgresql"
fi

# Check if Ollama is running
if ! pgrep -f "ollama" >/dev/null; then
    echo "⚠️  Warning: Ollama doesn't seem to be running."
    echo "   Please start Ollama with: ollama run llama3"
fi

echo "✅ Environment ready!"
echo "🚀 Launching Streamlit app..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the app"
echo ""

# Launch the Streamlit app
streamlit run receipt_app.py

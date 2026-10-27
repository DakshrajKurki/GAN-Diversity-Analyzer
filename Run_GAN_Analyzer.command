#!/bin/bash
# Double-click this file to run the GAN Image Quality and Diversity Analyzer.
# First run installs the libraries (needs internet), then the app opens in your browser.

cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif [ -x "$HOME/anaconda3/bin/python" ]; then
    PY="$HOME/anaconda3/bin/python"
else
    echo "Python 3 not found. Install it from https://www.python.org and try again."
    read -p "Press Enter to close..."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Setting up for the first time (this takes a few minutes)..."
    $PY -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
fi

# skip Streamlit's one-time email question
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "Starting the app... it will open in your browser."
echo "Keep this window open. Close it (or press Ctrl+C) to stop the app."
./.venv/bin/streamlit run app.py

#!/bin/bash 

install_requirements() {
  python3.10 -m venv audiobook_venv && \
  source ./audiobook_venv/bin/activate && pip install --upgrade pip && \
  pip install TTS pydub && \
  pip install torch && deactivate 2>&1 
}

install_requirements 2>&1 

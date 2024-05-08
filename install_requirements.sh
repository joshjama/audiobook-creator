#!/bin/bash 

install_requirements() {
  ./check_nvidea_gpu.sh  && \
  python3.10 -m venv audiobook_venv && \
  source ./audiobook_venv/bin/activate && pip install --upgrade pip && \
  pip install TTS pydub && \
  ./check_requirements_pip_installed.sh 2>&1  && \
  deactivate 2>&1 
}

install_requirements 2>&1 

#!/bin/bash 

create_audiobook() {
  source ./audiobook_venv/bin/activate && \
  python CreateAudiobook.py $1 $2 $3 2>&1 
}

create_audiobook $1 $2 $3 2>&1 

#!/bin/bash 

# Convert all wav files from a given folder into mp3 - files of a quality of 320 kb sound. 
convert_audios() {
  cd $1 && \
  for i in *.wav; do echo "Converting : " $i && \
  ffmpeg -i "$i" -ab 320k "${i%.*}.mp3"; done && \
  mkdir -p ./mp3 && mv *.mp3 ./mp3 
}

convert_audios $1 2>&1 

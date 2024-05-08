#!/bin/bash

# Überprüfen, ob eine CUDA-fähige GPU vorhanden ist
gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Wenn keine GPUs gefunden werden
if [ "$gpu_count" -eq 0 ]; then
  echo "Keine CUDA-fähige GPU gefunden."
  echo "No CUDA - capable GPU was found. "
  # Den Benutzer fragen, ob das Programm trotzdem installiert werden soll
  #read -p "Möchten Sie das Programm trotzdem installieren? (j/n): " antwort
  read -p "D you want to install Audiobook Creator anyway? (y/n): " antwort
  if [ "$antwort" = "n" ]; then
    #echo "Programm wird gestoppt."
    echo "Installation stopped"
    exit 1
  elif [ "$antwort" = "y" ]; then
    #echo "Programm wird fortgesetzt."
    echo "Continueing installation without CUDA-support. "
    # Fügen Sie hier den Code ein, um das Programm zu installieren
    exit 0 
  else
    echo "Ungültige Eingabe. Programm wird gestoppt."
    #exit 1
    echo "Invallid response. Stopping installation. "
  fi
else
  #echo "CUDA-fähige GPU gefunden. Anzahl der GPUs: $gpu_count"
  echo "Found CUDA - capable GPU. Number of GPUs: $gpu_count"
  # Fügen Sie hier den Code ein, um das Programm zu installieren
  exit 0 
fi


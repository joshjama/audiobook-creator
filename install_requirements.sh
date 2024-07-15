#!/bin/bash 

check_spacy_model_install() {
  
  source ./audiobook_venv/bin/activate && \
  if python -c "import spacy; spacy.load('de_core_news_sm')" && python -c "import spacy; spacy.load('en_core_web_sm')"; then
    #echo "Die spaCy-Modelle wurden erfolgreich installiert und sind kompatibel."
    echo "Successfully installed all spacy models. " 2>&1 
  else
    #echo "Es gab ein Problem beim Installieren oder Laden der spaCy-Modelle."
    echo "Error : There was a problem downloading the spacy models. Text will not be splitted proppaly. Please install a spacy model per language. "
  fi && \
  deactivate 2>&1 
}
install_spacy_models() {
  # Installieren von spaCy
  source ./audiobook_venv/bin/activate && \
  echo "Installing spaCy..." && \
  pip install spacy --timeout=10000000 && \

  # Herunterladen der spaCy-Modelle
  #echo "Lade das deutsche Modell 'de_core_news_sm' herunter..." && \
  echo "Downloading the german model for spacy. " && \
  python -m spacy download de_core_news_sm && \
  #echo "Lade das englische Modell 'en_core_web_sm' herunter..." && \
  echo "Downloading the english model for spacy. " && \
  python -m spacy download en_core_web_sm && \

  # Überprüfen, ob die Modelle erfolgreich installiert wurden
  check_spacy_model_install 
}

# Aufrufen der Funktion
#install_spacy_models

install_requirements() {
  ./check_nvidea_gpu.sh  && \
  python3.10 -m venv audiobook_venv && \
  source ./audiobook_venv/bin/activate && \
  pip install --upgrade pip --timeout=10000000 && \
  pip install TTS pydub spacy  langdetect flask ollama deepspeed --timeout=10000000 && \
  install_spacy_models && \
  ./check_requirements_pip_installed.sh && \
  check_spacy_model_install 2>&1 
  #deactivate 2>&1 
}

install_requirements 2>&1 

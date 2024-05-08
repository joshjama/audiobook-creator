#!/bin/bash

# Speichern Sie die Ausgabe von pip freeze in einer Variablen
source ./audiobook_venv/bin/activate && \
installed_packages=$(pip freeze) && \

# Lesen Sie den Inhalt der requirements.txt Datei
requirements=$(cat requirements.txt) && \

# Überprüfen Sie jedes Paket in der requirements.txt
for requirement in $requirements; do
  package_name=$(echo $requirement | cut -d'=' -f1)
  #if echo "$installed_packages" | grep -q "^$package_name=="; then
  if echo "$installed_packages" | grep -iq "^$package_name=="; then
    echo "Package $package_name is installed properly."
  else
    echo "Package $package_name was not installed properly. Please install it manually."
  fi
done


import sys 
import os
import re

def rename_audio_files(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return
    
    # Liste aller Dateien im Verzeichnis
    try:
        files = os.listdir(directory)
    except OSError as e:
        print(f"Error while reading directory: {e}")
        return
    
    # Filtere nur die Dateien, die dem Muster entsprechen
    #audio_files = [f for f in files if re.match(r'.*_\d+\.wav$', f)]
    audio_files = sorted([f for f in files if re.match(r'.*_\d+\.wav$', f)])
    
    if not audio_files:
        print("No matching audiofiles found.")
        return
    
    # Finde die maximale Zahl in den Dateinamen
    try:
        max_number = max(int(re.search(r'_(\d+)\.wav$', f).group(1)) for f in audio_files)
    except ValueError as e:
        print(f"Error while extracting numbers from filnames : {e}")
        return
    
    # Bestimme die Anzahl der Stellen der maximalen Zahl
    num_digits = len(str(max_number))
    
    for file in audio_files:
        # Extrahiere den Präfix und die Zahl
        match = re.match(r'(.*)_(\d+)\.wav$', file)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            
            # Erstelle den neuen Dateinamen mit führenden Nullen
            new_name = f"{prefix}_{number:0{num_digits}d}.wav"
            
            # Überprüfen, ob die neue Datei bereits existiert
            if os.path.exists(os.path.join(directory, new_name)):
                print(f"File {new_name} does already exist. Skipping.")
                continue
            
            # Benenne die Datei um
            try:
                os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
                print(f"Renamed: {file} -> {new_name}")
            except OSError as e:
                print(f"Error while renaming file {file} to {new_name}: {e}")

# Beispielaufruf
if __name__ == "__main__":  
  rename_audio_files(sys.argv[1]) 


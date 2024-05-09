import sys 
import os 
from TTS.api import TTS
import pydub 
from pydub import AudioSegment
from pydub.playback import play
import torch


## Usage : 
#$ python ./CreateAudiobook.py /PATH_TO_TEXT language your_books_name  

def create_audio_tts(text_file_path, LANGUAGE='en', book_name="Example_book") : 
  # Create audiobook directory 
  create_directory_from_book_name(book_name)
    # Get the device to use for TTS (use CUDA if available)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # Initialisierung des XTTS-Modells
  tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

  # Text, der in Sprache umgewandelt werden soll
  text = read_text_from_file(text_file_path)
  text_chunks = split_string_into_chunks(text, 1500)
  for index, chunk in enumerate(text_chunks) :

    # Umwandlung des Textes in Sprache und Speicherung in einer Datei
    output_path = book_name + "/" + book_name + f"_{index}.wav"
    text_to_speak = chunk 
    # Ersetzen Sie 'de_speaker_idx' durch den korrekten Index für einen deutschen Sprecher
    # und 'de_language_idx' durch den korrekten Index für die deutsche Sprache
    tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)

    # Abspielen der erzeugten Sprachdatei
    #audio = AudioSegment.from_wav("output.wav")

    # Abspielen der Audiodatei
    #play(audio)
  print("Audiobook generation finished") 


def read_text_from_file(file_path) : 
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return text
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Split the input text to chunks of 200 characters. 
def split_string_into_chunks(input_string, chunk_size) :
  # Initialisiere die Liste für die Chunks
  chunks = []
  current_chunk = ""
    
  # Teile den Text in Sätze
  sentences = input_string.split('. ')
    
  for sentence in sentences:
    # Füge einen Punkt hinzu, den wir beim Split entfernt haben, außer beim letzten Satz
    if sentence != sentences[-1]:
      sentence += '.'
        
      # Prüfe, ob der aktuelle Chunk plus der aktuelle Satz die Chunk-Größe überschreitet
      if len(current_chunk) + len(sentence) <= chunk_size:
        current_chunk += sentence + " "  # Füge den Satz zum aktuellen Chunk hinzu
      else:
        # Wenn der aktuelle Chunk voll ist, füge ihn zur Liste hinzu
        chunks.append(current_chunk.strip())
        current_chunk = sentence + " "  # Beginne einen neuen Chunk mit dem aktuellen Satz
    
  # Füge den letzten Chunk hinzu, falls vorhanden
  if current_chunk:
    chunks.append(current_chunk.strip())
    
  return chunks


# Example usage
#text = "Here is your example text that needs to be split into several chunks of 200 characters. " * 10  # A long string
#chunk_size = 200  # Define the chunk size as 200 characters

# Split the text and store it in a list
#chunks = split_string_into_chunks(text, chunk_size)

# Print the results
#for index, chunk in enumerate(chunks) :
#    print(f"Chunk {index + 1}:\n{chunk}\n")

# Check the length of each chunk to ensure they are 200 characters or less
#for index, chunk in enumerate(chunks) :
#    print(f"Length of chunk {index + 1}: {len(chunk)}")


def create_directory_from_book_name(book_name="Example_book") : 
    # Ersetze unerwünschte Zeichen im Buchnamen, um Probleme mit Dateisystembeschränkungen zu vermeiden
    # Hier werden beispielsweise Schrägstriche durch Unterstriche ersetzt
    sanitized_book_name = book_name.replace('/', '_').replace('\\', '_')
    
    # Pfad zum Verzeichnis, das erstellt werden soll
    directory_path = os.path.join(os.getcwd(), sanitized_book_name)
    
    # Überprüfe, ob das Verzeichnis bereits existiert
    if not os.path.exists(directory_path) :
        # Erstelle das Verzeichnis
        os.makedirs(directory_path)
        print(f"Verzeichnis '{directory_path}' wurde erfolgreich erstellt.")
    else:
        print(f"Das Verzeichnis '{directory_path}' existiert bereits.")

# Beispielverwendung
#book_name = "Mein tolles Buch"
#create_directory_from_book_name(book_name)





if __name__ == "__main__": 
  create_audio_tts(sys.argv[1], sys.argv[2], sys.argv[3]) 

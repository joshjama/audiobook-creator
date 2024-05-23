import sys 
import os 
from TTS.api import TTS
import pydub 
from pydub import AudioSegment
from pydub.playback import play
import torch
## Splitting Text to Sentences with spacy : 
import spacy
from langdetect import detect
from spacy.language import Language

## Usage : 
#$ python ./CreateAudiobook.py /PATH_TO_TEXT your_books_name  

# Language of the given Textfile : 
TEXT_LANGUAGE = "en" 

def load_model(language_code: str) -> Language:
  """Lädt das spaCy-Modell basierend auf dem Sprachcode."""
  if language_code == "de":
    return spacy.load("de_core_news_sm")
  elif language_code == "en":
    return spacy.load("en_core_web_sm")
  # Fügen Sie hier weitere Sprachen und Modelle hinzu
  else:
    raise ValueError(f"Kein Modell für die Sprache {language_code} verfügbar.")

def split_text_into_sentences(text: str, max_length: int = 250) -> list:
  """Teilt den Text in Sätze, mit Berücksichtigung der maximalen Länge."""
  language_code = detect(text)
  nlp = load_model(language_code)
  nlp.max_length = len(text) + 1
  try: 
    doc = nlp(text)
  except MemoryError:
    print("A MemoryError occurred. This usually means the system ran out of memory. Please try reducing the size of your input or closing other applications to free up memory.", file=sys.stderr )
    print("A MemoryError occurred while trying to process the text with spaCy. The text may be too long to fit into available RAM. Please try reducing the text size or increasing the available memory.", file=sys.stderr )
    sys.exit(1)  
    
  sentences = []
  current_chunk = ""
    
  for sent in doc.sents:
    if len(current_chunk) + len(sent.text) <= max_length or len(sent.text) > max_length:
      current_chunk += sent.text + " "
    else:
      sentences.append(current_chunk.strip())
      current_chunk = sent.text + " "

  if current_chunk:
    sentences.append(current_chunk.strip())
    
  return sentences

# Beispieltext
#text = "Hier ist ein langer Text, der in Sätze unterteilt werden soll. Dieser Text ist speziell dafür gedacht, um die Funktionsweise des Codes zu demonstrieren. Stellen Sie sicher, dass der Text in verschiedene Sprachen übersetzt werden kann, um die Multilingualität des Codes zu testen."

# Text in Sätze unterteilen
#sentences = split_text_into_sentences(text)

#print(sentences)




def create_audio_tts(text_file_path, LANGUAGE, book_name="Audiobook" ) : 
  # Create audiobook directory 
  create_directory_from_book_name(book_name)
    # Get the device to use for TTS (use CUDA if available)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # Initialisierung des XTTS-Modells
  tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

  # Text, der in Sprache umgewandelt werden soll
  text = read_text_from_file(text_file_path)
  if LANGUAGE == "en" or LANGUAGE == "de" : 
    LANGUAGE = detect(text)
    print("Detected language : ", LANGUAGE )
    text_chunks = split_text_into_sentences(text) 
  else : 
    print("Attension ! unsupported Language ! The text you insurted is not in one of the supported languages and will therefore not be splitted to sentences correctly. ")
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
  global TEXT_LANGUAGE 
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      text = file.read()
      language_code = detect(text)
      TEXT_LANGUAGE = language_code
      return text
  except FileNotFoundError:
    print("The file was not found.")
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
        #print(f"Verzeichnis '{directory_path}' wurde erfolgreich erstellt.")
        print(f"Directory '{directory_path}' was created successfully. ")
    else:
        #print(f"Das Verzeichnis '{directory_path}' existiert bereits.")
        print(f"Directory '{directory_path}' dos already exist. ")

# Beispielverwendung
#book_name = "Mein tolles Buch"
#create_directory_from_book_name(book_name)





if __name__ == "__main__": 
  if sys.argv[ len(sys.argv) - 1 ] != sys.argv[1] : 
    book_name = sys.argv[2] 
  else:  
    book_name = "Audiobook" 
  create_audio_tts(sys.argv[1], TEXT_LANGUAGE, book_name) 

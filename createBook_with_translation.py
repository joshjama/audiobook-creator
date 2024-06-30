import sys
import os
import traceback
import re
import argparse
from TTS.api import TTS
import pydub
from pydub import AudioSegment
from pydub.playback import play
import torch
from RenameAudios import *
import spacy
from langdetect import detect
from spacy.language import Language
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
from ollama import Client

## Usage : 
#$ python ./CreateAudiobook.py /PATH_TO_TEXT your_books_name  

# Language of the given Textfile : 
TEXT_LANGUAGE = "en" 
# States if the texts language was given by a cli argument. 
#text_language_set_externaly = False 

OLLAMA_URL = 'http://192.168.178.96:11434'

def load_model(language_code: str) -> Language:
    """Lädt das spaCy-Modell basierend auf dem Sprachcode."""
    if language_code == "de":
        return spacy.load("de_core_news_sm")
    elif language_code == "en":
        return spacy.load("en_core_web_sm")
    elif language_code == "fr":
        return spacy.load("fr_core_web_sm")
     # Fügen Sie hier weitere Sprachen und Modelle hinzu
    else:
        raise ValueError(f"Kein Modell für die Sprache {language_code} verfügbar.")

def determine_sentences_max_length(sentences):
    if not sentences:
        raise ValueError("Die Liste der Sätze darf nicht leer sein.")
        
    longest_sentence = ""
    max_length = 0

    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("Alle Elemente in der Liste müssen Zeichenketten sein.")
            
        sentence_length = len(sentence)
        if sentence_length > max_length:
            longest_sentence = sentence
            max_length = sentence_length

    print("The determined max Length inside the determine_max_length function is : " + str(max_length))
    return max_length 

def split_text_into_sentences(text: str, max_length_chunk: int = 500 ) -> list:
  """Teilt den Text in Sätze, mit Berücksichtigung der maximalen Länge."""
  language_code = TEXT_LANGUAGE  
  nlp = load_model(language_code)
  nlp.max_length = len(text) + 1
  try: 
    doc = nlp(text)
    print("Determining the chunklength to use for each TTS cycle. ")
    sentences_list = [] 
    sentences_list = [sent.text for sent in doc.sents]
    max_length_sentences = determine_sentences_max_length(sentences_list) 
    avg_length_sentences = int(sum(len(sent) for sent in sentences_list) / len(sentences_list)) 
    print(f"The longest sentence is {max_length_sentences} characters long.")
    print(f"The average sentence length is {avg_length_sentences} characters.")
        
    max_length_chunk = min(max_length_sentences * 2, avg_length_sentences * 3, 700)
    if  max_length_chunk < 500 and max_length_sentences <= 500 : 
      max_length_chunk = 500 
    elif max_length_sentences > 500 and max_length_sentences < 1500 : 
      max_length_chunk = max_length_sentences 
    elif max_length_sentences > 700 : 
      max_length_chunk = avg_length_sentences 
      while max_length_chunk < 500: 
        max_length_chunk += max_length_chunk 

    #print("The longest sentence is " + str(max_length_sentences) + " long. ") 
    #max_length_chunk = max_length_sentences 
    #print("The chunk_length is :" + str(max_length_chunk))
    print(f"The chunk length is: {max_length_chunk}")
  except MemoryError:
    print("A MemoryError occurred. This usually means the system ran out of memory. Please try reducing the size of your input or closing other applications to free up memory.", file=sys.stderr )
    print("A MemoryError occurred while trying to process the text with spaCy. The text may be too long to fit into available RAM. Please try reducing the text size or increasing the available memory.", file=sys.stderr )
    sys.exit(1)  
  except (ValueError, TypeError) as e:
    print("Error: Valueerror. Please ensure that your text is in the plane - text format. ")
    traceback.print_exc()
    print(f"Exception details: {e}")
    return []
    
  sentences = []
  current_chunk = ""
    
  for sent in doc.sents:
    if len(current_chunk) + len(sent.text) <= max_length_chunk or len(sent.text) > max_length_chunk:
      current_chunk += sent.text + " "
    else:
      sentences.append(current_chunk.strip())
      current_chunk = sent.text + " "

  if current_chunk:
    sentences.append(current_chunk.strip())
    
  # Cleaning sentences from unsupported characters : 
  print("Cleaning sentences from unusual characters. This may take some time ... ") 
  cleaned_sentences = clean_sentences(sentences) 
  sentences_with_rong_linebrakes = cleaned_sentences 
  # Clean linebrakes within sentences by saving linebrakes in the right possitions after !":" or at the end of a sentence. 
  print("Cleaning text from unwanted linebreaks. ") 
  sentences_with_lines = clean_line_breaks(sentences_with_rong_linebrakes) 
  print("Replaceing tabs \t with spaces. ")
  sentences_finished = remove_tabs_from_sentences(sentences_with_lines)
  sentences = sentences_finished 
  return sentences

# Beispieltext
#text = "Hier ist ein langer Text, der in Sätze unterteilt werden soll. Dieser Text ist speziell dafür gedacht, um die Funktionsweise des Codes zu demonstrieren. Stellen Sie sicher, dass der Text in verschiedene Sprachen übersetzt werden kann, um die Multilingualität des Codes zu testen."

# Text in Sätze unterteilen
#sentences = split_text_into_sentences(text)

#print(sentences)

def translate_text(text: str, target_language: str) -> str:
    """Translates text to the target language using a local Ollama instance."""
    client = Client(host=OLLAMA_URL)
    modelquery = f"Bitte übersetze den folgenden Text genau, ohne irgendwelche Änderungen oder das Hinzufügen von Wörtern, nach {target_language}:\n {text}\n Achte darauf, dass die Bedeutung und der Inhalt des Originaltextes vollständig und präzise erhalten bleiben. Verwende keine zusätzlichen Erklärungen, und vermeide jegliche Anpassungen, die nicht zwingend erforderlich sind, um den Text korrekt nach {target_language} zu übertragen."
    
    try:
        print(text)
        translation = ollama.generate(model="llama3", prompt=modelquery, stream=False)
        print ('##TRANSLATION: '+ translation['response'])
        return translation['response'] 
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_tts(tts, text_to_speak, output_path, language):
    try:
        print("The current output_path is : " + output_path)
        tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=language)
    except AssertionError:
        print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " instead.")
        print("The current output_path is : " + output_path)
        tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=TEXT_LANGUAGE)

def create_audio_tts(text_file_path, LANGUAGE, book_name="Audiobook", concatenate=False, translate=False, target_language="en", num_workers=4):
    # Create audiobook directory
    create_directory_from_book_name(book_name)
    # Get the device to use for TTS (use CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialisierung des XTTS-Modells
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Text, der in Sprache umgewandelt werden soll
    text = read_text_from_file(text_file_path)
    if text is None:
        print("Failed to read the text file. Exiting.")
        return

    language_detection_supported_for_textlanguage = True 
    if LANGUAGE == "en" or LANGUAGE == "de" : 
        LANGUAGE = TEXT_LANGUAGE 
        print("The Detected Main-language of your text is : ", LANGUAGE )
        print("Splitting your text into chunks, this may take some time ... ") 
        text_chunks = split_text_into_sentences(text) 
        language_detection_supported_for_textlanguage = True  
    else : 
        LANGUAGE = TEXT_LANGUAGE 
        print("Attention ! unsupported Language ! The text you inserted is not in one of the supported languages and will therefore not be splitted to sentences correctly. ")
        text_chunks = split_string_into_chunks(text, 1500)
        language_detection_supported_for_textlanguage = False

    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for index, chunk in enumerate(text_chunks):
            if translate:
                chunk = translate_text(chunk, target_language)
                LANGUAGE = target_language

            chunk_language = detect(chunk)
            if chunk_language != TEXT_LANGUAGE and language_detection_supported_for_textlanguage == True :
                print("Detected a different language in the current chunk.")
                print("Detected Chunk-Language : " + chunk_language)
                LANGUAGE = chunk_language
                chunk_chunks = split_text_into_sentences(chunk)
                for chunk_index, chunk_chunk in enumerate(chunk_chunks):
                    output_path = book_name + "/" + book_name + f"_{index}_{chunk_index}.wav"
                    tasks.append(executor.submit(generate_tts, tts, chunk_chunk, output_path, LANGUAGE))
                    text_to_speak = chunk_chunk 
                try: 
                    print("The current output_path is : " + output_path )
                    tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
                except AssertionError: 
                    print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " in stead." ) 
                    print("The current output_path is : " + output_path )
                    tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=TEXT_LANGUAGE)
                    index += 1
            else:
                if language_detection_supported_for_textlanguage == True :
                    LANGUAGE = chunk_language
                output_path = book_name + "/" + book_name + f"_{index}.wav"
                tasks.append(executor.submit(generate_tts, tts, chunk, output_path, LANGUAGE))
                

        for task in as_completed(tasks):
            task.result()  # To catch any exceptions that might have occurred

    if concatenate:
        concatenate_audio_files(book_name)
    print("Audiobook generation finished") 


def concatenate_audio_files(book_name):
    from pydub import AudioSegment

    # Get the directory where the audio files are stored
    directory_path = os.path.join(os.getcwd(), book_name)

    # List all audio files in the directory
    audio_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
    audio_files.sort()  # Ensure files are in the correct order

    # Initialize an empty AudioSegment
    combined = AudioSegment.empty()

    # Loop through each audio file and append it to the combined AudioSegment
    for audio_file in audio_files:
        file_path = os.path.join(directory_path, audio_file)
        audio_segment = AudioSegment.from_wav(file_path)
        combined += audio_segment

    # Export the combined audio file
    output_file_path = os.path.join(directory_path, f"{book_name}_combined.wav")
    combined.export(output_file_path, format="wav")
    print(f"Combined audio file created at: {output_file_path}")


def read_text_from_file(file_path):
    global TEXT_LANGUAGE, text_language_set_externaly 
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            language_code = detect(text)
            TEXT_LANGUAGE = language_code
            return text
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
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


# Clean the sentences fo your text from unusual and therefore not supported characters. 

def clean_sentences(sentences):
    """
    Replaces unusual or unsupported characters in a list of sentences with more common equivalents.
    Also removes line breaks within sentences but keeps line breaks after colons and sentence-ending punctuation.
    
    Args:
    sentences (list of str): List of sentences to be cleaned.
    
    Returns:
    list of str: Cleaned list of sentences.
    """
    replacements = {
        '«': '"',
        '»': '"',
        '„': '"',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '—': '-',
        '–': '-',
        '…': '...',
        '•': '-',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '©': '(c)',
        '®': '(r)',
        '™': '(tm)',
        '°': ' degrees',
        '±': '+/-',
        '×': 'x',
        '÷': '/',
        '∞': 'infinity',
        '≈': 'approx.',
        '≠': '!=',
        '≤': '<=',
        '≥': '>='
    }
    
    cleaned_sentences = []
    
    for sentence in sentences:
        try:
            # Replace unusual characters
            for old_char, new_char in replacements.items():
                #sentence = re.sub(re.escape(old_char), new_char, sentence)
                sentence = re.sub(re.escape(old_char) + r'(\w+)?' + re.escape(old_char), new_char, sentence)
            
            # Remove line breaks within sentences
            #sentence = re.sub(r'(?<![:.!?])\n', ' ', sentence)
            # Ensure line breaks after colons and sentence-ending punctuation are preserved
            #sentence = re.sub(r'([:])\n', r'\1\n', sentence)
            #sentence = re.sub(r'([.!?])\n', r'\1\n', sentence)
            
            cleaned_sentences.append(sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            cleaned_sentences.append(sentence)  # Ensure the original sentence is added if an error occurs
    
    return cleaned_sentences

# Clean linebreaks within sentences. 
def clean_line_breaks(sentences):
    """
    Removes line breaks within sentences but keeps line breaks after colons and sentence-ending punctuation.
    
    Args:
    sentences (list of str): List of sentences to be cleaned.
    
    Returns:
    list of str: Cleaned list of sentences.
    """
    cleaned_sentences = []
    
    for sentence in sentences:
        try:
            # Replace line breaks within sentences with a space
            sentence = re.sub(r'(?<![:.!?])\n', ' ', sentence)
            # Ensure line breaks after colons and sentence-ending punctuation are preserved
            sentence = re.sub(r'([:])\n', r'\1\n', sentence)
            sentence = re.sub(r'([.!?])\n', r'\1\n', sentence)
            cleaned_sentences.append(sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            cleaned_sentences.append(sentence)  # Ensure the original sentence is added if an error occurs
    
    return cleaned_sentences

# Replace \t with " " : 
def remove_tabs_from_sentences(sentences):
    """
    Entfernt alle Tabs aus den Sätzen und ersetzt sie durch Leerzeichen.
    
    :param sentences: Liste von Sätzen
    :return: Liste von Sätzen ohne Tabs
    """
    cleaned_sentences = []
    for sentence in sentences:
        try:
            cleaned_sentence = sentence.replace('\t', ' ')
            cleaned_sentences.append(cleaned_sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}")
            print(f"Exception details: {e}")
            traceback.print_exc()
            # Füge den ursprünglichen Satz hinzu, wenn eine Ausnahme auftritt
            cleaned_sentences.append(sentence)
    return cleaned_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audiobook from text file.")
    parser.add_argument("text_file_path", type=str, help="Path to the input text file.")
    parser.add_argument("book_name", type=str, help="Name of the audiobook.")
    parser.add_argument("--concatenate", action="store_true", help="Concatenate the generated audio files into one.")
    parser.add_argument("--translate", type=str, help="Translate the text to the specified language before TTS conversion.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel executions for the TTS function.")
    
    args = parser.parse_args()
    
    concatenate = args.concatenate
    translate = args.translate is not None
    target_language = args.translate if translate else "en"
    num_workers = args.num_workers
    
    create_audio_tts(args.text_file_path, TEXT_LANGUAGE, args.book_name, concatenate, translate, target_language, num_workers)
    audios_path = args.book_name + "/"
    rename_audio_files(audios_path)

import sys 
import os 
import traceback
import re
from TTS.api import TTS
import pydub 
from pydub import AudioSegment
from pydub.playback import play
import torch
from RenameAudios import * 
## Splitting Text to Sentences with spacy : 
import spacy
from langdetect import detect
from spacy.language import Language

## Usage : 
#$ python ./CreateAudiobook.py /PATH_TO_TEXT your_books_name  

# Language of the given Textfile : 
TEXT_LANGUAGE = "en" 
# States if the texts language was given by a cli argument. 
#text_language_set_externaly = False 

def load_model(language_code: str) -> Language:
  """Lädt das spaCy-Modell basierend auf dem Sprachcode."""
  if language_code == "de":
    return spacy.load("de_core_news_sm")
  elif language_code == "en":
    return spacy.load("en_core_web_sm")
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
    
  print("Cleaning sentences from unusual characters. This may take some time ... ") 
  cleaned_sentences = clean_sentences(sentences) 
  sentences_with_rong_linebrakes = cleaned_sentences 

  print("Cleaning text from unwanted linebreaks. ") 
  sentences_with_lines = clean_line_breaks(sentences_with_rong_linebrakes) 
  print("Replacing tabs \t with spaces. ")
  sentences_finished = remove_tabs_from_sentences(sentences_with_lines)
  sentences = sentences_finished 
  return sentences

def create_audio_tts(text_file_path, LANGUAGE, book_name="Audiobook" ) : 
  create_directory_from_book_name(book_name)
  log_file_path = os.path.join(book_name, "audio_files_log.txt")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
  text = read_text_from_file(text_file_path)
  language_detection_supported_for_textlanguage = True 
  if LANGUAGE == "en" or LANGUAGE == "de" : 
    LANGUAGE = TEXT_LANGUAGE 
    print("The Detected Main-language of your text is : ", LANGUAGE )
    print("Splitting your text into chunks, this may take some time ... ") 
    text_chunks = split_text_into_sentences(text) 
    language_detection_supported_for_textlanguage = True  
  else : 
    LANGUAGE = TEXT_LANGUAGE 
    print("Attension ! unsupported Language ! The text you inserted is not in one of the supported languages and will therefore not be split into sentences correctly. ")
    text_chunks = split_string_into_chunks(text, 200)
    language_detection_supported_for_textlanguage = False 
  index = 1  
  for l_index, chunk in enumerate(text_chunks) :
    chunk_language = detect(chunk) 
    if chunk_language != TEXT_LANGUAGE and language_detection_supported_for_textlanguage == True :  
      print("Detected a different language in the current chunk. ") 
      print("Detected Chunk-Language : " + chunk_language ) 
      LANGUAGE = chunk_language 
      chunk_chunks = split_text_into_sentences(chunk) 
      for chunk_index, chunk_chunk in enumerate(chunk_chunks) :
        output_path = book_name + "/" + book_name + f"_{index}.wav"
        text_to_speak = chunk_chunk 
        try: #AE1
          print("The current output_path is : " + output_path )
          tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
          with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"AE1: {output_path}\n")
          index += 1
        except AssertionError: 
          print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " instead." ) 
          print("An unexpected error was detected. The 400 token problem may have been the cause. ")
          traceback.print_exc()
          smaller_chunks = split_string_into_chunks(text_to_speak, 100) 
          print("This chunk is too long for xtts. Splitting into smaller chunks with split_string_into_chunks. This may cause irregular sentence splitting and may lead to blurring sounds. ")
          smaller_chunk_index = index 
          for small_chunk in smaller_chunks : 
            output_path = book_name + "/" + book_name + f"_{index}.wav"
            print("The current output-path is : " + output_path )
            text_to_speak = small_chunk 
            try: #AE2
              tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
              with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"AE2: {output_path}\n")
              index += 1
            except Exception as e:
              print("skipping") 
              continue 
          chunk_index = smaller_chunk_index 
          continue 
        except Exception as e:
          print("An unexpected error was detected. The 400 token problem may have been the cause. ")
          traceback.print_exc()
          continue 
    else: 
      if language_detection_supported_for_textlanguage == True : 
        LANGUAGE = chunk_language 
      output_path = book_name + "/" + book_name + f"_{index}.wav"
      text_to_speak = chunk 
      try: #BE1
        print("The current output_path is : " + output_path )
        tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
          log_file.write(f"BE1: {output_path}\n")
        index += 1
      except AssertionError: 
        print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " instead." ) 
        print("An unexpected error was detected. The 400 token problem may have been the cause. ")
        print("The current output_path is : " + output_path )
        traceback.print_exc()
        print("This chunk is too long for xtts. Splitting into smaller chunks with split_string_into_chunks. This may cause irregular sentence splitting and may lead to blurring sounds. ")
        smaller_chunks = split_string_into_chunks(text_to_speak, 100) 
        smaller_chunk_index = index 
        for small_chunk in smaller_chunks : 
          output_path = book_name + "/" + book_name + f"_{index}.wav"
          print("The current output-path is : " + output_path )
          text_to_speak = small_chunk 
          try: #BE2
            tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
              log_file.write(f"BE2: {output_path}\n")
            index += 1
          except Exception as e:
            print("skipping") 
            continue 
        continue 
      except Exception as e:
        print("An unexpected error was detected. The 400 token problem may have been the cause. ")
        traceback.print_exc()
        continue 
  print("Audiobook generation finished") 

def read_text_from_file(file_path) : 
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

def split_string_into_chunks(input_string, chunk_size) :
  chunks = []
  current_chunk = ""
  sentences = input_string.split('. ')
  for sentence in sentences:
    if sentence != sentences[-1]:
      sentence += '.'
      if len(current_chunk) + len(sentence) <= chunk_size:
        current_chunk += sentence + " "
      else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence + " "
  if current_chunk:
    chunks.append(current_chunk.strip())
  return chunks

def create_directory_from_book_name(book_name="Example_book") : 
    sanitized_book_name = book_name.replace('/', '_').replace('\\', '_')
    directory_path = os.path.join(os.getcwd(), sanitized_book_name)
    if not os.path.exists(directory_path) :
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created successfully. ")
    else:
        print(f"Directory '{directory_path}' does already exist. ")

def clean_sentences(sentences):
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
            for old_char, new_char in replacements.items():
                sentence = re.sub(re.escape(old_char) + r'(\w+)?' + re.escape(old_char), new_char, sentence)
            cleaned_sentences.append(sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def clean_line_breaks(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        try:
            sentence = re.sub(r'(?<![:.!?])\n', ' ', sentence)
            sentence = re.sub(r'([:])\n', r'\1\n', sentence)
            sentence = re.sub(r'([.!?])\n', r'\1\n', sentence)
            cleaned_sentences.append(sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def remove_tabs_from_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        try:
            cleaned_sentence = sentence.replace('\t', ' ')
            cleaned_sentences.append(cleaned_sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}")
            print(f"Exception details: {e}")
            traceback.print_exc()
            cleaned_sentences.append(sentence)
    return cleaned_sentences

if __name__ == "__main__": 
  if sys.argv[ len(sys.argv) - 1 ] != sys.argv[1] : 
    book_name = sys.argv[2] 
  else:  
    book_name = "Audiobook" 
  create_audio_tts(sys.argv[1], TEXT_LANGUAGE, book_name) 
  audios_path = book_name + "/" 
  rename_audio_files(audios_path)

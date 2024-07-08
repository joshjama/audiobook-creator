import sys 
import os 
import time 
import traceback
import re
from TTS.api import TTS
import pydub 
from pydub import AudioSegment
from pydub.playback import play
import torch
from RenameAudios import * 
from ConvertAudios import * 
## Splitting Text to Sentences with spacy : 
import spacy
from langdetect import detect
from spacy.language import Language
import ollama
from ollama import Client

#OLLAMA_URL = 'http://192.168.178.96:11434'
#OLLAMA_URL = 'http://127.0.0.1:59124'
OLLAMA_URL = 'http://localhost:11434'

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

def determine_paragraphs_max_length(paragraphs):
  if not paragraphs:
    raise ValueError("Die Liste der Sätze darf nicht leer sein.")
    
  longest_paragraph = ""
  max_length = 0

  for paragraph in paragraphs:
    if not isinstance(paragraph, str):
      raise TypeError("Alle Elemente in der Liste müssen Zeichenketten sein.")
        
    paragraph_length = len(paragraph)
    if paragraph_length > max_length:
      longest_paragraph = paragraph
      max_length = paragraph_length

  print("The determined max Length inside the determine_max_length function is : " + str(max_length))
  return max_length 

def split_text_into_paragraphs(text: str, max_length_chunk: int = 500 ) -> list: 
  """Teilt den Text in Sätze, mit Berücksichtigung der maximalen Länge."""
  language_code = TEXT_LANGUAGE  
  try: 
    nlp = load_model(language_code)
  except ValueError as VE : 
    sentences = split_string_into_chunks(text) 
    return sentences 
  nlp.max_length = len(text) + 1
  try: 
    doc = nlp(text)
    print("Determining the chunklength to use for each TTS cycle. ")
    paragraphs_list = [] 
    paragraphs_list = [sent.text for sent in doc.sents]
    max_length_paragraphs = determine_paragraphs_max_length(paragraphs_list) 
    avg_length_paragraphs = int(sum(len(sent) for sent in paragraphs_list) / len(paragraphs_list)) 
    print(f"The longest paragraph is {max_length_paragraphs} characters long.")
    print(f"The average paragraph length is {avg_length_paragraphs} characters.")
        
    max_length_chunk = min(max_length_paragraphs * 2, avg_length_paragraphs * 3, 700)
    if  max_length_chunk < 500 and max_length_paragraphs <= 500 : 
      max_length_chunk = 500 
    elif max_length_paragraphs > 500 and max_length_paragraphs < 1500 : 
      max_length_chunk = max_length_paragraphs 
    elif max_length_paragraphs > 700 : 
      max_length_chunk = avg_length_paragraphs 
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
    
  paragraphs = []
  current_chunk = ""
    
  for sent in doc.sents:
    if len(current_chunk) + len(sent.text) <= max_length_chunk or len(sent.text) > max_length_chunk:
      current_chunk += sent.text + " "
    else:
      paragraphs.append(current_chunk.strip())
      current_chunk = sent.text + " "

  if current_chunk:
    paragraphs.append(current_chunk.strip())
    
  print("Cleaning paragraphs from unusual characters. This may take some time ... ") 
  cleaned_paragraphs = clean_paragraphs(paragraphs) 
  paragraphs_with_rong_linebrakes = cleaned_paragraphs 

  print("Cleaning text from unwanted linebreaks. ") 
  paragraphs_with_lines = clean_line_breaks(paragraphs_with_rong_linebrakes) 
  print("Replacing tabs \t with spaces. ")
  paragraphs_finished = remove_tabs_from_paragraphs(paragraphs_with_lines)
  paragraphs = paragraphs_finished 
  return paragraphs

def create_audio_tts(text_file_path, LANGUAGE, book_name="Audiobook", speaker_idx='Claribel Dervla', translation_enabled=False, translate_to="German" ) : 
  create_directory_from_book_name(book_name)
  log_file_path = os.path.join(book_name, "audio_files_log.txt")
  text = read_text_from_file(text_file_path)
  language_detection_supported_for_textlanguage = True 
  if LANGUAGE == "en" or LANGUAGE == "de" : 
    LANGUAGE = TEXT_LANGUAGE 
    print("The Detected Main-language of your text is : ", LANGUAGE )
    print("Splitting your text into chunks, this may take some time ... ") 
    text_chunks = split_text_into_paragraphs(text) 
    language_detection_supported_for_textlanguage = True  
    # Testing : 
    #translation_enabled == True 
    if translation_enabled == True : 
      text_chunks_translated = translate_text_chunks(text_chunks, translate_to, True) 
      text_chunks = []
      text_chunks = text_chunks_translated 
      time.sleep(10)
      LANGUAGE = detect(text_chunks[0]) 
  else : 
    LANGUAGE = TEXT_LANGUAGE 
    print("Attension ! unsupported Language ! The text you inserted is not in one of the supported languages and will therefore not be split into paragraphs correctly. ")
    text_chunks = split_string_into_chunks(text, 200)
    language_detection_supported_for_textlanguage = False 
    if translation_enabled == True : 
      text_chunks_translated = translate_text_chunks(text_chunks, translate_to, True) 
      text_chunks = []
      text_chunks = text_chunks_translated 
      if translate_to == "German" : 
        LANGUAGE = "de" 
      else: 
        LANGUAGE = "en" 
  index = 0  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
  for l_index, chunk in enumerate(text_chunks) :
    chunk_language = detect(chunk) 
    if chunk_language != TEXT_LANGUAGE and language_detection_supported_for_textlanguage == True :  
      print("Detected a different language in the current chunk. ") 
      print("Detected Chunk-Language : " + chunk_language ) 
      LANGUAGE = chunk_language 
      chunk_chunks = split_text_into_paragraphs(chunk) 
      chunk_index = index 
      for l_chunk_index, chunk_chunk in enumerate(chunk_chunks) :
        output_path = book_name + "/" + book_name + f"_{chunk_index}.wav"
        text_to_speak = chunk_chunk 
        #if translation_enabled == True : 
          #text_to_speak =  translate_text(text_to_speak, translate_to) 
          #time.sleep(3)
        try: #AE1
          print("The current output_path is : " + output_path )
          tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
          with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
            #log_file.write(f"AE1: {output_path}\n")
          chunk_index += 1
        except AssertionError: 
          print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " instead." ) 
          print("An unexpected error was detected. The 400 token problem may have been the cause. ")
          traceback.print_exc()
          smaller_chunks = split_string_into_chunks(text_to_speak, 30) 
          print("This chunk is too long for xtts. Splitting into smaller chunks with split_string_into_chunks. This may cause irregular paragraph splitting and may lead to blurring sounds. ")
          smaller_chunk_index = index 
          for small_chunk in smaller_chunks : 
            output_path = book_name + "/" + book_name + f"_{chunk_index}.wav"
            print("The current output-path is : " + output_path )
            text_to_speak = small_chunk 
            #if translation_enabled == True : 
              #text_to_speak =  translate_text(text_to_speak, translate_to) 
              #time.sleep(3)
            try: #AE2
              tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
              with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
                #log_file.write(f"AE2: {output_path}\n")
              chunk_index += 1
            except Exception as e:
              print("skipping") 
              if LANGUAGE == "en" : 
                text_to_speak = "Sorry, this paragraphs seams to be to long. I am not able to read it. It may be that it is a long list of bibliographic information. Please check your spelling. " 
                try: #AE2
                  tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
                  with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
                    #log_file.write(f"AE2: {output_path}\n")
                  chunk_index += 1
                except Exception as e:
                  print("Skipping") 
                  continue 
              elif LANGUAGE == "de" : 
                text_to_speak = "Entschuldigung, Diese Sätze scheinen zu lang zur sein, um sie lesen zu können. Es könnte sich um eine sehr lange bibliographische Auflistung, oder ein Inhaltsverzeichnis handeln. Bitte achte außerdem darauf, das deine Zeichensetzung im Text korrekt ist. "
                try: #AE2
                  tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
                  with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
                    #log_file.write(f"AE2: {output_path}\n")
                  chunk_index += 1
                except Exception as e:
                  print("Skipping") 
                  continue 
              continue 
          chunk_index = smaller_chunk_index 
          continue 
        except Exception as e:
          print("An unexpected error was detected. The 400 token problem may have been the cause. ")
          traceback.print_exc()
          continue 
      index = chunk_index 
    else: 
      if language_detection_supported_for_textlanguage == True : 
        LANGUAGE = chunk_language 
      output_path = book_name + "/" + book_name + f"_{index}.wav"
      text_to_speak = chunk 
      #if translation_enabled == True : 
        #text_to_speak =  translate_text(text_to_speak, translate_to) 
        #time.sleep(3)
      try: #BE1
        print("The current output_path is : " + output_path )
        tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
          log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
          #log_file.write(f"BE1: {output_path}\n")
        index += 1
      except AssertionError: 
        print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " instead." ) 
        print("An unexpected error was detected. The 400 token problem may have been the cause. ")
        print("The current output_path is : " + output_path )
        traceback.print_exc()
        print("This chunk is too long for xtts. Splitting into smaller chunks with split_string_into_chunks. This may cause irregular paragraph splitting and may lead to blurring sounds. ")
        smaller_chunks = split_string_into_chunks(text_to_speak, 30) 
        smaller_chunk_index = index 
        for small_chunk in smaller_chunks : 
          output_path = book_name + "/" + book_name + f"_{index}.wav"
          print("The current output-path is : " + output_path )
          text_to_speak = small_chunk 
          #if translation_enabled == True : 
            #text_to_speak =  translate_text(text_to_speak, translate_to) 
            #time.sleep(3)
          try: #BE2
            tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
              log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
              #log_file.write(f"BE2: {output_path}\n")
            index += 1
          except Exception as e:
            print("skipping") 
            if LANGUAGE == "en" : 
              text_to_speak = "Sorry, this paragraphs seams to be to long. I am not able to read it. It may be that it is a long list of bibliographic information. Please check your spelling. " 
              try: #AE2
                tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                  log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
                  #log_file.write(f"AE2: {output_path}\n")
                index += 1
              except Exception as e:
                print("Skipping") 
                continue 
            elif LANGUAGE == "de" : 
              text_to_speak = "Entschuldigung, Diese Sätze scheinen zu lang zur sein, um sie lesen zu können. Es könnte sich um eine sehr lange bibliographische Auflistung, oder ein Inhaltsverzeichnis handeln. Bitte achte außerdem darauf, das deine Zeichensetzung im Text korrekt ist. "
              try: #AE2
                tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker=speaker_idx, language=LANGUAGE)
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                  log_file.write(f"AE1: {output_path}: \n{str(text_to_speak)} ")
                  #log_file.write(f"AE2: {output_path}\n")
                index += 1
              except Exception as e:
                print("Skipping") 
                continue 
            continue 
        continue 
      except Exception as e:
        print("An unexpected error was detected. The 400 token problem may have been the cause. ")
        traceback.print_exc()
        continue 
  folder = book_name + '/' 
  rename_audio_files(folder)
  convert_audios(folder)
  zip_mp3_files(folder, book_name) 
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

def split_string_into_chunks(input_string, chunk_size=500 ) :
  chunks = []
  current_chunk = ""
  paragraphs = input_string.split('. ')
  for paragraph in paragraphs:
    if paragraph != paragraphs[-1]:
      paragraph += '.'
      if len(current_chunk) + len(paragraph) <= chunk_size:
        current_chunk += paragraph + " "
      else:
        chunks.append(current_chunk.strip())
        current_chunk = paragraph + " "
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

def clean_paragraphs(paragraphs):
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
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        try:
            for old_char, new_char in replacements.items():
                #paragraph = re.sub(re.escape(old_char) + r'(\w+)?' + re.escape(old_char), new_char, paragraph)
                paragraph = paragraph.replace(old_char, new_char)
            cleaned_paragraphs.append(paragraph)
        except Exception as e:
            print(f"Error processing paragraph: {paragraph}. Error: {e}")
            cleaned_paragraphs.append(paragraph)
    return cleaned_paragraphs

def clean_line_breaks(paragraphs):
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        try:
            paragraph = re.sub(r'(?<![:.!?])\n', ' ', paragraph)
            paragraph = re.sub(r'([:])\n', r'\1\n', paragraph)
            paragraph = re.sub(r'([.!?])\n', r'\1\n', paragraph)
            cleaned_paragraphs.append(paragraph)
        except Exception as e:
            print(f"Error processing paragraph: {paragraph}. Error: {e}")
            cleaned_paragraphs.append(paragraph)
    return cleaned_paragraphs

def remove_tabs_from_paragraphs(paragraphs):
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        try:
            cleaned_paragraph = paragraph.replace('\t', ' ')
            cleaned_paragraphs.append(cleaned_paragraph)
        except Exception as e:
            print(f"Error processing paragraph: {paragraph}")
            print(f"Exception details: {e}")
            traceback.print_exc()
            cleaned_paragraphs.append(paragraph)
    return cleaned_paragraphs

def translate_text(text: str, target_language: str, model: str = "llama3") -> str:
    """Translates text to the target language using a local Ollama instance."""
    client = Client(host=OLLAMA_URL)
    modelquery = f"Translate the following text exactly, without any changes or adding words, to {target_language}:\n {text}\n Make sure that the meaning of the original text are fully and accurately preserved. Ensure that grammar and spelling of the translation are correct for language {target_language}. Use no additional explanations, and avoid any adjustments that are not strictly necessary to accurately translate the text into {target_language}. Do not add anything, just answer with the translated text. Exceptions from this rules ar not allowed in any case or for any reason! "
    
    try:
        print("Starting Translation. ")
        print("Translating your text into : " + target_language )
        print("Original Text : " )
        print(text)
        translation = ollama.generate(model=model, prompt=modelquery, stream=False)
        print ('##TRANSLATION: '+ translation['response'])
        del model
        #torch.cuda.empty_cache()
        return translation['response'] 
    except Exception as e:
        print(f"Translation error: {e}")
        traceback.print_exc()
        print(f"Exception details: {e}")
        return text

def translate_text_chunks(text_chunks, target_language, language_support ) -> list : 

  translated_chunks = []
  for chunk in text_chunks : 
    translated_chunk = translate_text(chunk, target_language ) 
    translated_chunks.append(translated_chunk)  

  translated_text = "" 
  for paragraph in translated_chunks : 
    translated_text = translated_text + paragraph  

  if language_support == True : 
    translated_text_chunks = split_text_into_paragraphs(translated_text )
  else : 
    translated_text_chunks = split_string_into_chunks(translated_text )
    
  return translated_text_chunks 


if __name__ == "__main__": 
  if sys.argv[ len(sys.argv) - 1 ] != sys.argv[1] : 
    book_name = sys.argv[2] 
  else:  
    book_name = "Audiobook" 
  create_audio_tts(sys.argv[1], TEXT_LANGUAGE, book_name) 
  #audios_path = book_name + "/" 
  #rename_audio_files(audios_path)

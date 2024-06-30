import sys
import os
import traceback
import re
import argparse
from TTS.api import TTS
import pydub
from pydub import AudioSegment
import torch
from RenameAudios import *
import spacy
from langdetect import detect
from spacy.language import Language
import ollama
from ollama import Client

# Global variable to set the text language
TEXT_LANGUAGE = "en"

OLLAMA_URL = 'http://192.168.178.96:11434'

def load_model(language_code: str) -> Language:
    """Load the spaCy model based on the language code."""
    if language_code == "de":
        return spacy.load("de_core_news_sm")
    elif language_code == "en":
        return spacy.load("en_core_web_sm")
    else:
        raise ValueError(f"No model available for language {language_code}.")

def determine_sentences_max_length(sentences):
    if not sentences:
        raise ValueError("The list of sentences cannot be empty.")
    
    longest_sentence = ""
    max_length = 0

    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("All elements in the list must be strings.")
        
        sentence_length = len(sentence)
        if sentence_length > max_length:
            longest_sentence = sentence
            max_length = sentence_length

    print("The determined max Length inside the determine_max_length function is : " + str(max_length))
    return max_length 

def split_text_into_sentences(text: str, max_length_chunk: int = 500) -> list:
    """Split text into sentences, considering the maximum length."""
    language_code = TEXT_LANGUAGE  
    nlp = load_model(language_code)
    nlp.max_length = len(text) + 1
    try: 
        doc = nlp(text)
        print("Determining the chunk length to use for each TTS cycle.")
        sentences_list = [sent.text for sent in doc.sents]
        max_length_sentences = determine_sentences_max_length(sentences_list) 
        avg_length_sentences = int(sum(len(sent) for sent in sentences_list) / len(sentences_list)) 
        print(f"The longest sentence is {max_length_sentences} characters long.")
        print(f"The average sentence length is {avg_length_sentences} characters.")
        
        max_length_chunk = min(max_length_sentences * 2, avg_length_sentences * 3, 700)
        if max_length_chunk < 500 and max_length_sentences <= 500: 
            max_length_chunk = 500 
        elif max_length_sentences > 500 and max_length_sentences < 1500: 
            max_length_chunk = max_length_sentences 
        elif max_length_sentences > 700: 
            max_length_chunk = avg_length_sentences 
            while max_length_chunk < 500: 
                max_length_chunk += max_length_chunk 

        print(f"The chunk length is: {max_length_chunk}")
    except MemoryError:
        print("A MemoryError occurred. Please try reducing the size of your input or closing other applications to free up memory.", file=sys.stderr)
        sys.exit(1)  
    except (ValueError, TypeError) as e:
        print("Error: ValueError. Please ensure that your text is in plain text format.")
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
    
    print("Cleaning sentences from unusual characters. This may take some time ...") 
    cleaned_sentences = clean_sentences(sentences) 

    print("Cleaning text from unwanted line breaks.") 
    sentences_with_lines = clean_line_breaks(cleaned_sentences) 
    print("Replacing tabs with spaces.")
    sentences_finished = remove_tabs_from_sentences(sentences_with_lines)
    sentences = sentences_finished 
    return sentences

def translate_text(text: str, target_language: str, model: str = "llama") -> str:
    """Translates text to the target language using a local Ollama instance."""
    client = Client(host=OLLAMA_URL)
    modelquery = f"Translate the following text exactly, without any changes or adding words, to {target_language}:\n {text}\n Make sure that the meaning of the original text are fully and accurately preserved. Use no additional explanations, and avoid any adjustments that are not strictly necessary to accurately translate the text into {target_language}. Do not add anything, just answer with the translated text."
    
    try:
        print(text)
        translation = ollama.generate(model=model, prompt=modelquery, stream=False)
        print ('##TRANSLATION: '+ translation['response'])
        return translation['response'] 
    except Exception as e:
        print(f"Translation error: {e}")
        return text
    
def create_audio_tts(text_file_path, LANGUAGE, book_name="Audiobook", translate=False, target_language="en", model="llama"): 
    """Create audio files from text using TTS."""
    create_directory_from_book_name(book_name)
    log_file_path = os.path.join(book_name, "audio_files_log.txt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    text = read_text_from_file(text_file_path)
    language_detection_supported_for_textlanguage = True 
    if LANGUAGE == "en" or LANGUAGE == "de": 
        LANGUAGE = TEXT_LANGUAGE 
        print("The detected main language of your text is:", LANGUAGE)
        print("Splitting your text into chunks, this may take some time ...") 
        text_chunks = split_text_into_sentences(text) 
    else: 
        LANGUAGE = TEXT_LANGUAGE 
        print("Attention! Unsupported language! The text you inserted is not in one of the supported languages and will therefore not be split into sentences correctly.")
        text_chunks = split_string_into_chunks(text, 200)
        language_detection_supported_for_textlanguage = False 
    
    index = 1  
    for l_index, chunk in enumerate(text_chunks):
        if translate:
                chunk = translate_text(chunk, target_language, model=model)
                LANGUAGE = target_language

        chunk_language = detect(chunk) 
        if chunk_language != TEXT_LANGUAGE and language_detection_supported_for_textlanguage:  
            print("Detected a different language in the current chunk.") 
            print("Detected chunk language:", chunk_language) 
            LANGUAGE = chunk_language 
            chunk_chunks = split_text_into_sentences(chunk) 
            for chunk_index, chunk_chunk in enumerate(chunk_chunks):
                output_path = os.path.join(book_name, f"{book_name}_{index}.wav")
                text_to_speak = chunk_chunk 
                try:  # AE1
                    print("The current output path is:", output_path)
                    tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"AE1: {output_path}\n")
                    index += 1
                except AssertionError: 
                    print("The detected chunk language is not supported by the xtts v2 model. Using", TEXT_LANGUAGE, "instead.") 
                    print("An unexpected error was detected. The 400 token problem may have been the cause.")
                    traceback.print_exc()
                    smaller_chunks = split_string_into_chunks(text_to_speak, 100) 
                    for small_chunk in smaller_chunks:
                        output_path = os.path.join(book_name, f"{book_name}_{index}.wav")
                        print("The current output path is:", output_path)
                        text_to_speak = small_chunk 
                        try:  # AE2
                            tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
                            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                                log_file.write(f"AE2: {output_path}\n")
                            index += 1
                        except Exception as e:
                            print("Skipping")
                            continue 
                    continue 
                except Exception as e:
                    print("An unexpected error was detected. The 400 token problem may have been the cause.")
                    traceback.print_exc()
                    continue 
        else: 
            if language_detection_supported_for_textlanguage:
                LANGUAGE = chunk_language 
            output_path = os.path.join(book_name, f"{book_name}_{index}.wav")
            text_to_speak = chunk 
            try:  # BE1
                print("The current output path is:", output_path)
                tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"BE1: {output_path}\n")
                index += 1
            except AssertionError: 
                print("The detected chunk language is not supported by the xtts v2 model. Using", TEXT_LANGUAGE, "instead.") 
                print("An unexpected error was detected. The 400 token problem may have been the cause.")
                traceback.print_exc()
                smaller_chunks = split_string_into_chunks(text_to_speak, 100) 
                for small_chunk in smaller_chunks:
                    output_path = os.path.join(book_name, f"{book_name}_{index}.wav")
                    print("The current output path is:", output_path)
                    text_to_speak = small_chunk 
                    try:  # BE2
                        tts.tts_to_file(text=text_to_speak, file_path=output_path, speaker='Claribel Dervla', language=LANGUAGE)
                        with open(log_file_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"BE2: {output_path}\n")
                        index += 1
                    except Exception as e:
                        print("Skipping")
                        continue 
                continue 
            except Exception as e:
                print("An unexpected error was detected. The 400 token problem may have been the cause.")
                traceback.print_exc()
                continue 
    print("Audiobook generation finished.") 

def read_text_from_file(file_path): 
    """Read text from a file and detect its language."""
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

def split_string_into_chunks(input_string, chunk_size):
    """Split a string into chunks of specified size."""
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

def create_directory_from_book_name(book_name="Example_book"):
    """Create a directory based on the book name."""
    sanitized_book_name = book_name.replace('/', '_').replace('\\', '_')
    directory_path = os.path.join(os.getcwd(), sanitized_book_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def clean_sentences(sentences):
    """Clean sentences from unusual characters."""
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
    """Clean unwanted line breaks from sentences."""
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
    """Remove tabs from sentences and replace them with spaces."""
    cleaned_sentences = []
    for sentence in sentences:
        try:
            cleaned_sentence = sentence.replace('\t', ' ')
            cleaned_sentences.append(cleaned_sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def concatenate_audio_files(book_name):
    """Concatenate all audio files in the book directory into a single file."""
    directory_path = os.path.join(os.getcwd(), book_name)
    audio_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
    audio_files.sort()

    combined = AudioSegment.empty()

    for audio_file in audio_files:
        file_path = os.path.join(directory_path, audio_file)
        audio_segment = AudioSegment.from_wav(file_path)
        combined += audio_segment

    output_file_path = os.path.join(directory_path, f"{book_name}_combined.wav")
    combined.export(output_file_path, format="wav")
    print(f"Combined audio file created at: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an audiobook from a text file using TTS.")
    parser.add_argument('text_file_path', type=str, help='Path to the text file.')
    parser.add_argument('book_name', type=str, nargs='?', default='Audiobook', help='Name of the audiobook (optional).')
    parser.add_argument('--language', type=str, default='en', help='Language of the text file (default: en).')
    parser.add_argument('--concat', action='store_true', help='Concatenate all audio files into a single file.')
    parser.add_argument("--translate", type=str, help="Translate the text to the specified language before TTS conversion.")
    parser.add_argument('--model', type=str, default='llama', help='Model to use for translation (default: llama).')
    
    args = parser.parse_args()
    
    #concatenate = args.concat
    translate = args.translate is not None
    target_language = args.translate if translate else args.language

    TEXT_LANGUAGE = args.language
    create_audio_tts(args.text_file_path, TEXT_LANGUAGE, args.book_name, translate, target_language, model=args.model)
    
    audios_path = os.path.join(args.book_name, "")
    rename_audio_files(audios_path)
    
    if args.concat:
        concatenate_audio_files(args.book_name)

# Example usage:
# python CreateAudiobook.py /path/to/textfile.txt My_Audiobook --language en --concat --model llama

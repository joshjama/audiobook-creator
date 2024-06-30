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
import spacy
from langdetect import detect
from spacy.language import Language
import argparse

TEXT_LANGUAGE = "en"
DEFAULT_MAX_CHUNK_LENGTH = 400

def load_model(language_code: str) -> Language:
    if language_code == "de":
        return spacy.load("de_core_news_sm")
    elif language_code == "en":
        return spacy.load("en_core_web_sm")
    else:
        raise ValueError(f"No model available for language {language_code}.")

def determine_sentences_max_length(sentences):
    if not sentences:
        raise ValueError("Sentence list cannot be empty.")
    
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

def split_long_sentence(sentence, max_length):
    """Split a long sentence into shorter sentences, each no longer than max_length tokens."""
    words = sentence.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(' '.join(current_chunk + [word])) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def split_text_into_sentences(text: str, max_length_chunk: int = DEFAULT_MAX_CHUNK_LENGTH) -> list:
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
        
        # Print the longest sentence
        longest_sentence = max(sentences_list, key=len)
        #print(f"The longest sentence is: {longest_sentence}")

        # Verwende die minimale Länge zwischen berechneter max_length_chunk und DEFAULT_MAX_CHUNK_LENGTH
        max_length_chunk = min(DEFAULT_MAX_CHUNK_LENGTH, max_length_chunk)

        print(f"The chunk length is: {max_length_chunk}")
    except MemoryError:
        print("A MemoryError occurred. Please try reducing the size of your input or closing other applications to free up memory.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, TypeError) as e:
        print("Error: Valueerror. Please ensure that your text is in the plain-text format.")
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

    final_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > max_length_chunk:
            final_sentences.extend(split_long_sentence(sentence, max_length_chunk))
        else:
            final_sentences.append(sentence)

    print("Cleaning sentences from unusual characters. This may take some time ...")
    cleaned_sentences = clean_sentences(final_sentences)
    sentences_with_rong_linebrakes = cleaned_sentences
    print("Cleaning text from unwanted linebreaks.")
    sentences_with_lines = clean_line_breaks(sentences_with_rong_linebrakes)
    print("Replacing tabs with spaces.")
    sentences_finished = remove_tabs_from_sentences(sentences_with_lines)
    sentences = sentences_finished
    return sentences

def create_audio_tts(text_file_path, LANGUAGE, book_name="Audiobook", concatenate=False, max_chunk_length=DEFAULT_MAX_CHUNK_LENGTH):
    create_directory_from_book_name(book_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    text = read_text_from_file(text_file_path)
    if text is None:
        print("Failed to read the text file. Exiting.")
        return

    if LANGUAGE not in ["en", "de"]:
        print("Attention! Unsupported language! The text you inserted is not in one of the supported languages.")
        text_chunks = split_string_into_chunks(text, max_chunk_length)
    else:
        LANGUAGE = TEXT_LANGUAGE
        print("The Detected Main-language of your text is : ", LANGUAGE)
        print("Splitting your text into chunks, this may take some time ...")
        text_chunks = split_text_into_sentences(text, max_chunk_length)

    index = 0
    for chunk in text_chunks:
        chunk_language = detect(chunk)
        if chunk_language not in ["en", "de"]:
            chunk_language = TEXT_LANGUAGE

        sentences = []
        current_chunk = ""
        for sentence in chunk.split('. '):
            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    sentences.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            sentences.append(current_chunk.strip())

        # Ensure chunks are within max length
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > max_chunk_length:
                final_sentences.extend(split_long_sentence(sentence, max_chunk_length))
            else:
                final_sentences.append(sentence)

        for sentence in final_sentences:
            if sentence.strip():  # Ensure the chunk is not empty
                print(f"Sentence length: {len(sentence)} / max_chunk_length: {max_chunk_length}")
                output_path = os.path.join(book_name, f"{book_name}_{index}.wav")
                try:
                    print("The current output_path is : " + output_path)
                    tts.tts_to_file(text=sentence, file_path=output_path, speaker='Claribel Dervla', language=chunk_language)
                except AssertionError:
                    print("The detected Chunk-Language is not supported by the xtts v2 model. Using " + TEXT_LANGUAGE + " instead.")
                    tts.tts_to_file(text=sentence, file_path=output_path, speaker='Claribel Dervla', language=TEXT_LANGUAGE)
                index += 1

    if concatenate:
        concatenate_audio_files(book_name)
    print("Audiobook generation finished")

def concatenate_audio_files(book_name):
    from pydub import AudioSegment

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

def read_text_from_file(file_path):
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
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def split_string_into_chunks(input_string, chunk_size):
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
    sanitized_book_name = book_name.replace('/', '_').replace('\\', '_')
    directory_path = os.path.join(os.getcwd(), sanitized_book_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

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

def main():
    parser = argparse.ArgumentParser(description="Create an audiobook from a text file.")
    parser.add_argument('text_file', type=str, help="Path to the text file.")
    parser.add_argument('book_name', type=str, help="Name of the audiobook.")
    parser.add_argument('--concatenate', action='store_true', help="Concatenate all audio files into one.")
    parser.add_argument('--max_chunk_length', type=int, default=DEFAULT_MAX_CHUNK_LENGTH, help="Maximum length of each chunk in tokens.")
    #python CreateAudiobook.py /path/to/your/textfile.txt YourBookName --concatenate --max_chunk_length 400

    args = parser.parse_args()

    create_audio_tts(args.text_file, TEXT_LANGUAGE, args.book_name, args.concatenate, args.max_chunk_length)
    audios_path = args.book_name + "/"
    rename_audio_files(audios_path)

if __name__ == "__main__":
    main()

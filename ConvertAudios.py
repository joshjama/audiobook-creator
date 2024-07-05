import os
import zipfile
import sys
from pydub import AudioSegment

def convert_audios(folder):
    os.chdir(folder)
    for file in os.listdir():
        if file.endswith(".wav"):
            print("Converting:", file)
            input_file = file
            output_file = file[:-4] + ".mp3"
            audio = AudioSegment.from_wav(input_file)
            audio.export(output_file, format="mp3", bitrate="320k")
    
    os.makedirs("mp3", exist_ok=True)
    for file in os.listdir():
        if file.endswith(".mp3"):
            os.rename(file, os.path.join("mp3", file))


def zip_mp3_files(folder, book_name):
    mp3_folder = "./mp3" 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    mp3_folder = script_directory + "/" + book_name + "/" + "mp3" 
    print(script_directory) 
    print(mp3_folder) 
    #mp3_folder = os.path.join("mp3")
    #os.chdir(folder)
    os.chdir(mp3_folder)
    #os.chdir(mp3_folder)
    #"/mp3"
    if not os.path.exists(mp3_folder):
        print("No 'mp3' folder found.")
        return
    
    zip_filename = book_name + '.zip'
    with zipfile.ZipFile(zip_filename, "w") as zip_file:
        for file in os.listdir(mp3_folder):
            if file.endswith(".mp3"):
                file_path = os.path.join(mp3_folder, file)
                zip_file.write(file_path, os.path.basename(file_path))
                #zip_file.write(os.path.join(mp3_folder, file))
    
    print(f"ZIP file '{zip_filename}' created successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder> <book_name> ")
        sys.exit(1)
    
    book_name = sys.argv[2] 
    folder = sys.argv[1]
    convert_audios(folder)
    zip_mp3_files(folder, book_name) 
    


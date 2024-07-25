import os
import sys
import threading
import traceback
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from CreateAudiobook import create_audio_tts, TEXT_LANGUAGE
from RenameAudios import * 

app = Flask(__name__)
UPLOAD_FOLDER = 'Downloads'
AUDIOBOOKS_FOLDER = '.'
ALLOWED_EXTENSIONS = {'txt'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIOBOOKS_FOLDER, exist_ok=True)

# Global variable to toggle to use the gpu or cpu to generate the tts - audios.. 
use_gpu = True 

# Global variable to control translation
translation_enabled = False
# Supported languages
languages_supported = ['German', 'English']

# Global variable for the selected language
translate_to = 'German'

# Voice_temperature defines if results are more deterministic or more creative. 
# a0.0 is as much deterministic while 1.0 ist as much creative. 
voice_temperature = 0.85 

# Te speaker idxs to select for xtts : 
speaker_idxs = [
'Claribel Dervla', 
'Daisy Studious', 
'Gracie Wise', 
'Tammie Ema', 
'Alison Dietlinde', 
'Ana Florence', 
'Annmarie Nele', 
'Asya Anara', 
'Brenda Stern', 
'Gitta Nikolina', 
'Henriette Usha', 
'Sofia Hellen', 
'Tammy Grit', 
'Tanja Adelina', 
'Vjollca Johnnie', 
'Andrew Chipper', 
'Badr Odhiambo', 
'Dionisio Schuyler', 
'Royston Min', 
'Viktor Eka', 
'Abrahan Mack', 
'Adde Michal', 
'Baldur Sanjin', 
'Craig Gutsy', 
'Damien Black', 
'Gilberto Mathias', 
'Ilkin Urbano', 
'Kazuhiko Atallah', 
'Ludvig Milivoj', 
'Suad Qasim', 
'Torcull Diarmuid', 
'Viktor Menelaos', 
'Zacharie Aimilios', 
'Nova Hogarth', 
'Maja Ruoho', 
'Uta Obando', 
'Lidiya Szekeres', 
'Chandra MacFarland', 
'Szofi Granger', 
'Camilla Holmström', 
'Lilya Stainthorpe', 
'Zofija Kendrick', 
'Narelle Moon', 
'Barbora MacLean', 
'Alexandra Hisakawa', 
'Alma María', 
'Rosemary Okafor', 
'Ige Behringer', 
'Filip Traverse', 
'Damjan Chapman', 
'Wulf Carlevaro', 
'Aaron Dreschner', 
'Kumar Dahl', 
'Eugenio Mataracı', 
'Ferran Simen', 
'Xavier Hayasaka', 
'Luis Moray', 
'Marcos Rudaski'
] 

#Speaker to select via drop-down : 
speaker_idx = ''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global speaker_idx, translate_to, translation_enabled, voice_temperature, use_gpu 
    

    
    if 'toggle' in request.form:
        toggle_value = request.form['toggle']
        
        if toggle_value == 'on':
            translation_enabled = True
        elif toggle_value == 'off':
            translation_enabled = False
        else:
            # Invalid toggle value, set an error message
            error_message = 'Invalid toggle value.'
            #return render_template('index.html', languages=languages_supported, selected_language=translate_to, translation_enabled=translation_enabled, error=error_message)
    else:
        # Toggle parameter missing in the form, set an error message
        error_message = 'Toggle parameter missing.'
        #return render_template('index.html', languages=l
        #return render_template_string('index.html', languages=languages_supported, selected_language=translate_to, translation_enabled=translation_enabled, error=error_message)

    # toggle gpu : 
    if 'toggle_gpu' in request.form:
        toggle_value = request.form['toggle_gpu']
        
        if toggle_value == 'on':
            use_gpu = True
            print("Toggled GPU-Usage. ")
            print("Current gpu-ussage : " + str(use_gpu) ) 
        elif toggle_value == 'off':
            use_gpu = False
            print("Toggled GPU-Usage. ")
            print("Current gpu-ussage : " + str(use_gpu) ) 
        else:
            # Invalid toggle value, set an error message
            error_message = 'Invalid toggle value.'
            #return render_template('index.html', languages=languages_supported, selected_language=translate_to, translation_enabled=translation_enabled, error=error_message)
    if request.method == 'POST':

       if 'submit_voice_temperature' in request.form:
          voice_temperature = float(request.form['voice_temperature'])
       # Check if the language parameter is present in the form
       if 'language' in request.form:
          selected_language = request.form['language']
            
          # Check if the selected language is in the list of supported languages
          if selected_language in languages_supported:
             translate_to = selected_language
          else:
             # Invalid language selected, set an error message
             error_message = 'Invalid language selected.'
             #return render_template('index.html', languages=languages_supported, selected_language=translate_to, error=error_message)
       else:
            # Language parameter missing in the form, set an error message
          error_message = 'Language parameter missing.'
          #return render_template('index.html', languages=languages_supported, selected_language=translate_to, error=error_message)


    if request.method == 'POST':
       try:
          selected_speaker = request.form['speaker']
          if selected_speaker in speaker_idxs:
             speaker_idx = selected_speaker
          else:
              raise ValueError('Ungültiger Sprecher ausgewählt')
       except KeyError:
          pass  # Kein Sprecher ausgewählt
       except ValueError as e:
          print(f"Fehler: {str(e)}")

    audiobook_folders = [f for f in os.listdir('.') if os.path.isdir(os.path.join('.', f))]
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Audiobook-Creator</title>
        <script>
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
            }
        </script>
    </head>
    <body>
    <h1>Language Selection</h1>
    
    {% if error %}
        <p style="color: red;">{{ error }}</p>
    {% endif %}
    
    <form method="POST">
        <select name="language">
            {% for language in languages %}
                <option value="{{ language }}" {% if language == selected_language %}selected{% endif %}>{{ language }}</option>
            {% endfor %}
        </select>
        <input type="submit" value="Select">
    </form>
    
    <p>Selected Language: {{ selected_language }}</p>
<form method="POST">
    <div>
        <input type="radio" id="toggle_on" name="toggle" value="on" {% if translation_enabled %}checked{% endif %}>
        <label for="toggle_on">Translation On</label>
    </div>
    <div>
        <input type="radio" id="toggle_off" name="toggle" value="off" {% if not translation_enabled %}checked{% endif %}>
        <label for="toggle_off">Translation Off</label>
    </div>
    <input type="submit" value="Toggle Translation">
</form>

<p>Translation Enabled: {{ translation_enabled }}</p>

    <h1>Select Speaker</h1>
    <form method="post">
        <label for="speaker-select">Speaker:</label>
        <select name="speaker" id="speaker-select">
            <option value="">Please select a speaker.</option>
            {% for speaker in speaker_idxs %}
                <option value="{{ speaker }}" {% if speaker == selected_speaker %}selected{% endif %}>{{ speaker }}</option>
            {% endfor %}
        </select>
        <br><br>
        <input type="submit" value="Select">
    </form>

        <h1>Text to Audiobook</h1>
        <form action="/upload" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <label for="text">Enter text:</label><br>
            <textarea id="text" name="text" rows="10" cols="50"></textarea><br><br>
            <label for="file">Or upload a text file:</label><br>
            <input type="file" id="file" name="file"><br><br>
            <label for="audiobook_name">Audiobook Name:</label><br>
            <input type="text" id="audiobook_name" name="audiobook_name"><br><br>
            <input type="submit" value="Submit">
        </form>

        <form method="POST">
          <div>
            <label for="voice_temperature">Voice-Temperature:</label>
            <input type="text" id="voice_temperature" name="voice_temperature" value="{{ voice_temperature }}" placeholder="default is 0.85">
          </div>
          <button type="submit" name="submit_voice_temperature">Submit</button>
        </form>

<form method="POST">
    <div>
        <input type="radio" id="toggle_on_gpu" name="toggle_gpu" value="on" {% if use_gpu %}checked{% endif %}>
        <label for="toggle_on_gpu">GPU-Usage On</label>
    </div>
    <div>
        <input type="radio" id="toggle_off_gpu" name="toggle_gpu" value="off" {% if not use_gpu %}checked{% endif %}>
        <label for="toggle_off_gpu">GPU_Usage Off</label>
    </div>
    <input type="submit" value="Toggle_gpu">
</form>

        <h2>Generated Audio Files</h2>
        <ul>
            {% for folder in audiobook_folders %}
                <li>
                    <a href="{{ url_for('list_files', folder=folder) }}">{{ folder }}</a>
                </li>
            {% endfor %}
        </ul>
    </body>
    </html>
    ''', audiobook_folders=audiobook_folders, os=os, AUDIOBOOKS_FOLDER=AUDIOBOOKS_FOLDER, speaker_idxs=speaker_idxs, selected_speaker=speaker_idx, languages=languages_supported, selected_language=translate_to, translation_enabled=translation_enabled, use_gpu=use_gpu )

@app.route('/files/<folder>')
def list_files(folder):
    files = os.listdir(os.path.join(AUDIOBOOKS_FOLDER, folder))
    files.sort()  # Sort files to ensure correct order
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{{ folder }}</title>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var audioElements = document.querySelectorAll('audio');
                for (let i = 0; i < audioElements.length - 1; i++) {
                    audioElements[i].addEventListener('ended', function() {
                        audioElements[i + 1].play();
                    });
                }
            });
        </script>
    </head>
    <body>
        <h1>{{ folder }}</h1>
        <ul>
            {% for file in files %}
                <li>
                    <audio controls>
                        <source src="{{ url_for('download_file', folder=folder, filename=file) }}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <a href="{{ url_for('download_file', folder=folder, filename=file) }}" target="_blank">{{ file }}</a>
                </li>
            {% endfor %}
        </ul>
        <a href="{{ url_for('index') }}">Back to home</a>
    </body>
    </html>
    ''', folder=folder, files=files)

def list_files_old_old(folder):
    files = os.listdir(os.path.join(AUDIOBOOKS_FOLDER, folder))
    files.sort()  # Sort files to ensure correct order
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{{ folder }}</title>
    </head>
    <body>
        <h1>{{ folder }}</h1>
        <ul>
            <audio controls>
                {% for file in files %}
                    <source src="{{ url_for('download_file', folder=folder, filename=file) }}" type="audio/wav">
                {% endfor %}
                Your browser does not support the audio element.
            </audio>
            {% for file in files %}
                <li>
                    <a href="{{ url_for('download_file', folder=folder, filename=file) }}" target="_blank">{{ file }}</a>
                </li>
            {% endfor %}
        </ul>
        <a href="{{ url_for('index') }}">Back to home</a>
    </body>
    </html>
    ''', folder=folder, files=files)

def list_files_old(folder):
    files = os.listdir(os.path.join(AUDIOBOOKS_FOLDER, folder))
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{{ folder }}</title>
    </head>
    <body>
        <h1>{{ folder }}</h1>
        <ul>
            {% for file in files %}
                <li>
                    <audio controls>
                        <source src="{{ url_for('download_file', folder=folder, filename=file) }}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <a href="{{ url_for('download_file', folder=folder, filename=file) }}" target="_blank">{{ file }}</a>
                </li>
            {% endfor %}
        </ul>
        <a href="{{ url_for('index') }}">Back to home</a>
    </body>
    </html>
    ''', folder=folder, files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    print('Submit button pressed')
    text = request.form.get('text')
    audiobook_name = request.form.get('audiobook_name')
    audiobook_name = audiobook_name.replace(" ", "") 
    audiobook_name = audiobook_name.replace(".", "") 
    if not audiobook_name:
        audiobook_name = 'Audiobook_' + str(len(os.listdir(AUDIOBOOKS_FOLDER)) + 1)
    audiobook_folder = os.path.join(AUDIOBOOKS_FOLDER, audiobook_name)
    audiobook_name = audiobook_folder.replace("/", "") 
    file_path = None

    if text and text.strip():
        file_path = os.path.join(UPLOAD_FOLDER, 'input.txt')
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f'Text saved to {file_path}')
        except Exception as e:
            print(f'Error saving text: {str(e)}')
            return f'Error saving text: {str(e)}', 500
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return 'No selected file', 400
        if not allowed_file(file.filename):
            print('File type not allowed')
            return 'File type not allowed', 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(file_path)
            print(f'File saved to {file_path}')
        except Exception as e:
            print(f'Error saving file: {str(e)}')
            return f'Error saving file: {str(e)}', 500
    else:
        print('No text or file provided')
        return 'No text or file provided', 400

    thread = threading.Thread(target=create_audio_tts_with_logging, args=(file_path, TEXT_LANGUAGE, audiobook_folder, speaker_idx, translation_enabled, translate_to, voice_temperature, use_gpu ))
    thread.start()
    print('File uploaded and processing started')

    return redirect(url_for('index'))

def create_audio_tts_with_logging(file_path, text_language, audiobook_folder, speaker_idx='Claribel Dervla', translation_enabled=False, translate_to='German', voice_temperature=0.85, use_gpu=True ):
    os.makedirs(audiobook_folder, exist_ok=True)
    try:
        print(f'Starting audiobook creation for {file_path}')
        create_audio_tts(file_path, text_language, audiobook_folder, speaker_idx, translation_enabled, translate_to, voice_temperature, use_gpu )
        book_name = audiobook_folder.replace("/", "")  
        book_name = book_name.replace(".", "") 
        audios_path = book_name + "/" 
        #rename_audio_files(audios_path) 
        print(f'Audiobook creation completed for {file_path}')
    except Exception as e:
        print(f'Error during audiobook creation: {str(e)}')
        traceback.print_exc()
    finally:
        try:
            os.remove(file_path)
            print(f'File {file_path} deleted after processing')
        except Exception as e:
            print(f'Error deleting file {file_path}: {str(e)}')

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    try:
        print(f'Starting download for {filename} in folder {folder}')
        response = send_from_directory(os.path.join(AUDIOBOOKS_FOLDER, folder), filename, mimetype='audio/wav')
        print(f'Download completed for {filename} in folder {folder}')
        return response
    except Exception as e:
        print(f'Error downloading file: {str(e)}')
        return f'Error downloading file: {str(e)}', 500

#@app.route('/toggle_translation', methods=['POST'])
#def toggle_translation():



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


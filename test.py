from ollama import Client
import ollama

TEXT_LANGUAGE = "en"
OLLAMA_URL = 'http://192.168.178.96:11434'

text='Ein kleiner Heinzelmann tanzt Walzer!'
target_language='fr'

client = Client(host=OLLAMA_URL)
modelquery = f"You are a highly sophisticated translation expert. Translate the given text marked by TEXT: to the target language marked by LANGUAGE:\nTEXT:\n{text}\nLANGUAGE:\n{target_language}"
    
translation = ollama.generate(model="phi3", prompt=modelquery, stream=False)
print("##TRANSLATION")
print(translation['response'])
#print ('##TRANSLATION: '+''.join([t['text'] for t in translation]))

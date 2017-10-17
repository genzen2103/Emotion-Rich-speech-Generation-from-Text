import subprocess

def textToWav(text,file_name):
   subprocess.call(["espeak", "-w"+file_name+".wav", text])


file_handle = open('chpt36_full_text.txt',"r")
text = file_handle.readline()
file_handle.close()
textToWav(text,'chpt36_tts')





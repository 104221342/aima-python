import os
import google.generativeai as genai # type: ignore

genai.configure(api_key='AIzaSyBz9k9zazyymUrwoIOm3Unj_nf-rsH-8lQ')

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

while True:
    prompt = input("Ask me anything: ")
    if (prompt == "exit"):
        break
    response = chat.send_message(prompt, stream=True)
    for chunk in response:
        if chunk.text:
          print(chunk.text)
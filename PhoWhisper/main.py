from transformers import pipeline
transcriber = pipeline("automatic-speech-recognition", model="/work/PhoWhisper-medium")
output = transcriber("2023-01-11-2022-0338954101-14.41.mp3")['text']

print(output)

# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("2023-01-11-2022-0338954101-14.41.mp3")
# print(result["text"])



# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("audio.mp3")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)
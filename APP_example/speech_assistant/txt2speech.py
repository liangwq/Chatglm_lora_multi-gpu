from TTS.api import TTS
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="我爱你中国，中国山河壮丽",
                file_path="output.wav",
                speaker_wav="/root/autodl-tmp/TTS/TTS/female.wav",
                )

# generate speech by cloning a voice using custom settings
tts.tts_to_file(text="我爱你中国，中国山河壮丽",
                file_path="output1.wav",
                speaker_wav="/root/autodl-tmp/TTS/TTS/female.wav",
            
                decoder_iterations=30)


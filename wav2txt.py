import paddlespeech
from paddlespeech.cli.asr.infer import ASRExecutor

def wav2txt(wav_path: str):
    asr = ASRExecutor()
    result = asr(audio_file=wav_path, model='conformer_wenetspeech')
    return result
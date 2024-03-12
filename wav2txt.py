import paddlespeech
from paddlespeech.cli.asr.infer import ASRExecutor

def wav2txt(wav_path: str, lang: str='zh', sample_rate: int=16000):
    asr = ASRExecutor()
    if lang=='zh':
        model = 'conformer_wenetspeech'
    elif lang=='en':
        model = 'conformer_librispeech'
    result = asr(audio_file=wav_path, 
                 model=model,
                 lang=lang,
                 sample_rate=sample_rate)
    return result
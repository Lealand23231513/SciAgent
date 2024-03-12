from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor

def wav2txt(wav_path: str, lang: str='zh', sample_rate: int=16000):
    asr = ASRExecutor()
    if lang=='zh':
        model = 'conformer_wenetspeech'
    elif lang=='en':
        model = 'transformer_librispeech'
    result = asr(
        audio_file=wav_path, 
        model=model,
        lang=lang,
        sample_rate=sample_rate)
    return result

def wav2txt_client(wav_path: str, server_ip: str='127.0.0.1', port: int=8090, lang: str='zh', sample_rate: int=16000):
    asrclient_executor = ASRClientExecutor()
    result = asrclient_executor(
        input=wav_path,
        server_ip=server_ip,
        port=port,
        sample_rate=sample_rate,
        lang=lang,
        audio_format="wav")
    return result
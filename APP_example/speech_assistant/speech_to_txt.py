import torch

import gradio as gr
import yt_dlp as youtube_dl
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

import tempfile
import os

MODEL_NAME = "openai/whisper-large-v2"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model="/root/autodl-tmp/whisper-large-v2",#MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return  text


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise gr.Error(str(err))
    
    file_length = info["duration_string"]
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
    
    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)
    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
    
    if file_length_s > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
        raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")
    
    ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(str(err))


def yt_transcribe(yt_url, task, max_filesize=75.0):
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(yt_url, filepath)
        with open(filepath, "rb") as f:
            inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

    return html_embed_str, text


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V2: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="upload", type="filepath", optional=True, label="Audio file"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V2: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe")
    ],
    outputs=["html", "text"],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V2: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the checkpoint"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe video files of"
        " arbitrary length."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe], ["Microphone", "Audio file", "YouTube"])

demo.launch(enable_queue=True,server_port= 6006)

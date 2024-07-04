import ssl
from flask import Flask, request, jsonify
from flask_cors import CORS
from pytube import YouTube, exceptions
import os
import sqlite3
from contextlib import closing
import torch
import soundfile as sf
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import ffmpeg
import warnings
from datetime import timedelta
import json
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

app = Flask(__name__)
CORS(app)

ssl._create_default_https_context = ssl._create_unverified_context

DATABASE = 'database.db'

# Whisper Large-v3 모델 로드
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="eager"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps="word",
    torch_dtype=torch_dtype,
    device=device,
)

def connect_db():
    return sqlite3.connect(DATABASE)

def init_db():
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with closing(connect_db()) as db:
        with open(schema_path, mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello, World!"

def format_time_srt(seconds):
    milliseconds = int(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_subtitles(segments):
    subtitles = []
    buffer = ""
    start_time = None
    end_time = None
    subtitle_index = 1

    for segment in segments:
        text = segment['text']
        timestamp = segment['timestamp']

        if start_time is None:
            start_time = timestamp[0]

        buffer += text + " "
        end_time = timestamp[1] if timestamp[1] is not None else timestamp[0] + 1

        if len(buffer) > 20 or any(punct in text for punct in ['.', ',', '!', '?']):
            srt_subtitle = f"{subtitle_index}\n{format_time_srt(start_time)} --> {format_time_srt(end_time)}\n{buffer.strip()}\n"
            subtitles.append(srt_subtitle)
            buffer = ""
            start_time = None
            subtitle_index += 1

    if buffer:
        srt_subtitle = f"{subtitle_index}\n{format_time_srt(start_time)} --> {format_time_srt(end_time)}\n{buffer.strip()}\n"
        subtitles.append(srt_subtitle)

    return subtitles

def save_subtitles_to_srt(subtitles, base_filename):
    srt_filepath = os.path.join('srtfiles', base_filename + '.srt')
    with open(srt_filepath, 'w', encoding='utf-8') as srt_file:
        srt_file.writelines(subtitles)
    print(f"Saved subtitles to {srt_filepath}")

def normalize_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    query_params.pop('t', None)
    normalized_query = urlencode(query_params, doseq=True)
    normalized_url = urlunparse(parsed_url._replace(query=normalized_query))
    return normalized_url


def split_audio(input_filepath, segment_length=60):
    # 1분 간격(60초)으로 오디오 파일 분할
    output_path = os.path.join('downloads', 'splits')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_template = os.path.join(output_path, 'split_%03d.wav')
    ffmpeg.input(input_filepath).output(output_template, acodec='pcm_s16le', ar='16k', ac=1, segment_time=segment_length, f='segment').run()
    return output_path


def process_audio_segments(segment_folder, language):
    for segment_file in sorted(os.listdir(segment_folder)):
        segment_path = os.path.join(segment_folder, segment_file)
        print(f"Processing {segment_path}")
        result = pipe(segment_path, generate_kwargs={"language": language})
        subtitles = generate_subtitles(result['chunks'])
        base_filename = os.path.splitext(segment_file)[0]  # 확장자 없는 파일명 추출
        save_subtitles_to_srt(subtitles, base_filename)

def merge_subtitles(srt_folder, final_filename):
    all_subtitles = []
    for srt_file in sorted(os.listdir(srt_folder)):
        srt_path = os.path.join(srt_folder, srt_file)
        with open(srt_path, 'r', encoding='utf-8') as file:
            all_subtitles.extend(file.readlines())
            all_subtitles.append('\n')  # 각 SRT 파일 사이에 공백 줄 추가

    final_srt_path = os.path.join('srtfiles', final_filename + '.srt')
    with open(final_srt_path, 'w', encoding='utf-8') as final_srt_file:
        final_srt_file.writelines(all_subtitles)
    print(f"Final subtitles saved to {final_srt_path}")

    # 기존의 임시 SRT 파일들 삭제
    for srt_file in os.listdir(srt_folder):
        os.remove(os.path.join(srt_folder, srt_file))
    os.rmdir(srt_folder)  # 빈 폴더 삭제

@app.route('/api/download_audio', methods=['POST'])
def download_audio():
    data = request.get_json()
    url = data.get('url')
    language = data.get('language', 'ko')  # 기본값으로 한국어로 설정

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    normalized_url = normalize_url(url)

    try:
        with connect_db() as db:
            cursor = db.cursor()
            cursor.execute('SELECT subtitles FROM downloads WHERE url = ?', (normalized_url,))
            if cursor.fetchone():
                return jsonify({'message': 'URL already exists in the database', 'url': normalized_url}), 200

        yt = YouTube(url)
        if yt.age_restricted:
            return jsonify({'error': 'Video is age restricted and cannot be downloaded without logging in'}), 403

        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            return jsonify({'error': 'No audio stream available'}), 404

        safe_title = ''.join([c for c in yt.title if c.isalnum() or c == ' ']).strip()
        filename = f"{safe_title}.mp3"
        filepath = os.path.join('downloads', filename)
        audio_stream.download(output_path='downloads', filename=filename)

        converted_filepath = os.path.join('downloads', f"{safe_title}_converted.wav")
        ffmpeg.input(filepath).output(converted_filepath, format='wav', acodec='pcm_s16le', ar='16k', ac=1).run()

        splits_folder = split_audio(converted_filepath)
        process_audio_segments(splits_folder, language)

        # Cleanup
        os.remove(filepath)
        os.remove(converted_filepath)
        for segment_file in os.listdir(splits_folder):
            os.remove(os.path.join(splits_folder, segment_file))
        os.rmdir(splits_folder)

        return jsonify({'message': 'Audio processed and subtitles generated successfully'})

    except Exception as e:
        app.logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    if not os.path.exists('srtfiles'):
        os.makedirs('srtfiles')
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)
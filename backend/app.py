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
    for i, segment in enumerate(segments):
        start_time = segment['timestamp'][0]
        end_time = segment['timestamp'][1] if segment['timestamp'][1] is not None else start_time + 1
        text = segment['text']
        srt_subtitle = f"{i+1}\n{format_time_srt(start_time)} --> {format_time_srt(end_time)}\n{text}\n"
        subtitles.append(srt_subtitle)
    return subtitles

def normalize_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    query_params.pop('t', None)
    normalized_query = urlencode(query_params, doseq=True)
    normalized_url = urlunparse(parsed_url._replace(query=normalized_query))
    return normalized_url

@app.route('/api/download_audio', methods=['POST'])
def download_audio():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    normalized_url = normalize_url(url)

    try:
        with connect_db() as db:
            cursor = db.cursor()
            cursor.execute('SELECT subtitles FROM downloads WHERE url = ?', (normalized_url,))
            row = cursor.fetchone()
            if row:
                subtitles = json.loads(row[0])
                return jsonify(
                    {'message': 'URL already exists in the database', 'url': normalized_url, 'subtitles': subtitles}), 200

        try:
            yt = YouTube(url)
            if yt.age_restricted:
                return jsonify({'error': 'Video is age restricted and cannot be downloaded without logging in'}), 403
        except exceptions.AgeRestrictedError:
            return jsonify({'error': 'Video is age restricted and cannot be accessed without logging in'}), 403

        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            return jsonify({'error': 'No audio stream available'}), 404

        # 파일명을 YouTube 제목으로 설정하고, 파일명에서 허용되지 않는 문자를 제거
        safe_title = ''.join([c for c in yt.title if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
        filename = f"{safe_title}.mp3"
        filepath = os.path.join('downloads', filename)

        app.logger.info(f"Downloading audio")
        audio_stream.download(output_path='downloads', filename=filename)
        app.logger.info(f"End Downloaded")

        # 오디오 파일 변환 (단일 채널, 모델이 지원하는 형식으로)
        app.logger.info(f"Saving audio to {filepath}")
        converted_filepath = os.path.join('downloads', f"{safe_title}_converted.wav")
        ffmpeg.input(filepath).output(converted_filepath, format='wav', acodec='pcm_s16le', ar='16k', ac=1).run()

        app.logger.info(f"Audio saved to {converted_filepath}")

        # 파일이 존재하는지 확인
        if not os.path.exists(converted_filepath):
            app.logger.error(f"Converted file does not exist: {converted_filepath}")
            return jsonify({'error': f"Converted file does not exist: {converted_filepath}"}), 500

        # Whisper Large-v3 모델로 자막 생성
        app.logger.info(f"Starting transcription with Whisper model")
        result = pipe(converted_filepath)
        subtitles = generate_subtitles(result['chunks'])
        app.logger.info(f"End Transcribed with Whisper model")

        # 데이터베이스에 자막 저장
        with connect_db() as db:
            cursor = db.cursor()
            cursor.execute('SELECT id FROM downloads WHERE url = ?', (normalized_url,))
            row = cursor.fetchone()
            if row:
                db.execute('UPDATE downloads SET subtitles = ?, timestamp = CURRENT_TIMESTAMP WHERE url = ?', (json.dumps(subtitles), normalized_url))
            else:
                db.execute('INSERT INTO downloads(url, subtitles) VALUES (?, ?)', (normalized_url, json.dumps(subtitles)))
            db.commit()

        # 파일 삭제
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(converted_filepath):
            os.remove(converted_filepath)
        app.logger.info(f"Deleted files: {filepath} and {converted_filepath}")

        return jsonify(
            {'message': 'Audio downloaded successfully', 'filename': filename, 'subtitles': subtitles})
    except Exception as e:
        app.logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    if not os.path.exists(DATABASE):
        init_db()

    app.run(debug=True)

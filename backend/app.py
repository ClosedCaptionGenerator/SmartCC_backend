import ssl
from flask import Flask, request, jsonify
from flask_cors import CORS
from pytube import YouTube
import os
import sqlite3
from contextlib import closing
import torch
import torchaudio
import soundfile as sf
from transformers import pipeline
import ffmpeg
import warnings
from datetime import timedelta
import json
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

app = Flask(__name__)
CORS(app)

ssl._create_default_https_context = ssl._create_unverified_context

DATABASE = 'database.db'
pipe = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")


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


def format_time(seconds):
    return str(timedelta(seconds=seconds)).split(".")[0]


def generate_subtitles(predictions, segment_duration):
    subtitles = []
    for i, prediction in enumerate(predictions):
        label = prediction['label']
        start_time = format_time(i * segment_duration)
        end_time = format_time((i + 1) * segment_duration)
        subtitles.append(f"{i + 1}\n{start_time} --> {end_time}\n[{label}]\n")
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

        yt = YouTube(url)
        if yt.age_restricted:
            return jsonify({'error': 'Video is age restricted and cannot be downloaded without logging in'}), 403

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

        # 오디오 데이터를 PyTorch 텐서로 변환
        app.logger.info(f"Starting non-speech analysis with AST model")
        waveform, sample_rate = sf.read(converted_filepath)
        waveform = torch.tensor(waveform).unsqueeze(0)

        segment_duration = 10
        num_samples_per_segment = sample_rate * segment_duration

        total_length = waveform.shape[1]
        if total_length < num_samples_per_segment:
            padding = num_samples_per_segment - total_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), 'constant', 0)

        num_segments = waveform.shape[1] // num_samples_per_segment
        ast_predictions = []

        for i in range(num_segments):
            start = i * num_samples_per_segment
            end = (i + 1) * num_samples_per_segment
            segment = waveform[:, start:end]

            # pipeline으로 모델 사용
            prediction = pipe(segment.squeeze().numpy(), sampling_rate=sample_rate)
            ast_predictions.append(prediction[0])

        subtitles = generate_subtitles(ast_predictions, segment_duration)
        app.logger.info(f"End Transcribed with AST model")

        os.remove(filepath)
        os.remove(converted_filepath)
        app.logger.info(f"Deleted files: {filepath} and {converted_filepath}")

        with connect_db() as db:
            cursor = db.cursor()
            cursor.execute('SELECT id FROM downloads WHERE url = ?', (normalized_url,))
            row = cursor.fetchone()
            if row:
                db.execute('UPDATE downloads SET subtitles = ?, timestamp = CURRENT_TIMESTAMP WHERE url = ?', (json.dumps(subtitles), normalized_url))
            else:
                db.execute('INSERT INTO downloads(url, subtitles) VALUES (?, ?)', (normalized_url, json.dumps(subtitles)))
            db.commit()

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

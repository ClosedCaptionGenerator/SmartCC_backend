from flask import Flask, request, jsonify
from flask_cors import CORS
from pytube import YouTube
import os

app = Flask(__name__)
CORS(app)  # CORS 설정


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello, World!"


@app.route('/api/download_audio', methods=['POST'])
def download_audio():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        yt = YouTube(url)
        if yt.age_restricted:
            return jsonify({'error': 'Video is age restricted and cannot be downloaded without logging in'}), 403

        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            return jsonify({'error': 'No audio stream available'}), 404

        # 파일명을 YouTube 제목으로 설정하고, 파일명에서 허용되지 않는 문자를 제거합니다.
        safe_title = ''.join([c for c in yt.title if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
        filename = f"{safe_title}.mp3"

        audio_stream.download(output_path='downloads', filename=filename)  # mp3 형식으로 다운로드
        return jsonify({'message': 'Audio downloaded successfully', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    app.run(debug=True)

document.addEventListener('DOMContentLoaded', () => {
  const downloadButton = document.getElementById('downloadButton');
  const youtubeUrlInput = document.getElementById('youtubeUrl');
  const languageSelect = document.getElementById('languageSelect');
  const statusMessage = document.getElementById('statusMessage');
  const astSubtitlesTextarea = document.getElementById('astSubtitles');

  downloadButton.addEventListener('click', () => {
    const youtubeUrl = youtubeUrlInput.value;
    const selectedLanguage = languageSelect.value;

    if (!youtubeUrl) {
      statusMessage.textContent = 'Please enter a YouTube URL';
      return;
    }

    fetch('http://127.0.0.1:5000/api/download_audio', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: youtubeUrl, language: selectedLanguage })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        statusMessage.textContent = `Error: ${data.error}`;
      } else {
        statusMessage.textContent = `Audio downloaded successfully: ${data.filename}`;
        astSubtitlesTextarea.value = data.subtitles.join('\n');
      }
    })
    .catch(error => {
      console.error('Error:', error);
      statusMessage.textContent = 'An error occurred while downloading the audio';
    });
  });
});

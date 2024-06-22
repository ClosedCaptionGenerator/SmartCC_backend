document.addEventListener('DOMContentLoaded', () => {
  const downloadButton = document.getElementById('downloadButton');
  const youtubeUrlInput = document.getElementById('youtubeUrl');
  const statusMessage = document.getElementById('statusMessage');
  const astSubtitlesTextarea = document.getElementById('astSubtitles');

  downloadButton.addEventListener('click', () => {
    const youtubeUrl = youtubeUrlInput.value;
    if (!youtubeUrl) {
      statusMessage.textContent = 'Please enter a YouTube URL';
      return;
    }

    fetch('https://web-flask.jollyground-483c02b7.westus2.azurecontainerapps.io/api/download_audio', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: youtubeUrl })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        statusMessage.textContent = `Error: ${data.error}`;
      } else {
        statusMessage.textContent = `Audio downloaded successfully: ${data.filename}`;
        const astSubtitles = data.ast_subtitles.map(sub => `${sub}`).join('\n');
        astSubtitlesTextarea.value = astSubtitles;
      }
    })
    .catch(error => {
      console.error('Error:', error);
      statusMessage.textContent = 'An error occurred while downloading the audio';
    });
  });
});

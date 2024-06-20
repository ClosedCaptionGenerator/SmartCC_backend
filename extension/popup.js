document.addEventListener('DOMContentLoaded', () => {
  const downloadButton = document.getElementById('downloadButton');
  const youtubeUrlInput = document.getElementById('youtubeUrl');
  const statusMessage = document.getElementById('statusMessage');

  downloadButton.addEventListener('click', () => {
    const youtubeUrl = youtubeUrlInput.value;
    if (!youtubeUrl) {
      statusMessage.textContent = 'Please enter a YouTube URL';
      return;
    }

    fetch('http://127.0.0.1:5000/api/download_audio', {
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
      }
    })
    .catch(error => {
      console.error('Error:', error);
      statusMessage.textContent = 'An error occurred while downloading the audio';
    });
  });
});

document.addEventListener('DOMContentLoaded', () => {
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const sessionNameInput = document.getElementById('sessionName');
    const statusDiv = document.getElementById('status');
    const transcriptDiv = document.getElementById('transcript');

    let mediaRecorder;
    let audioChunks = [];

    recordButton.addEventListener('click', async () => {
        if (sessionNameInput.value.trim() === '') {
            alert('Please enter a session name.');
            return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                socket.emit('audio_chunk', event.data);
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            recordButton.disabled = true;
            stopButton.disabled = false;
            sessionNameInput.disabled = true;
            statusDiv.textContent = 'Status: Recording...';
            transcriptDiv.innerHTML = ''; // Clear previous transcript
            console.log('Recording started');
        };

        mediaRecorder.onstop = () => {
            recordButton.disabled = false;
            stopButton.disabled = true;
            sessionNameInput.disabled = false;
            statusDiv.textContent = 'Status: Stopped. Processing final audio...';
            console.log('Recording stopped');

            // Combine all chunks into a single blob
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

            // Send final audio data and session name to the server
            socket.emit('stop_recording', {
                'session_name': sessionNameInput.value,
                'audio_data': audioBlob
            });

            audioChunks = []; // Reset for next recording
        };

        // Start recording and send chunks every 2 seconds
        mediaRecorder.start(2000);
    });

    stopButton.addEventListener('click', () => {
        mediaRecorder.stop();
    });

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });

    socket.on('transcript_update', (data) => {
        const entry = document.createElement('div');
        entry.classList.add('transcript-entry');

        const speaker = document.createElement('span');
        speaker.classList.add('speaker');
        speaker.textContent = `${data.speaker}: `;

        const text = document.createElement('span');
        text.textContent = data.text;

        entry.appendChild(speaker);
        entry.appendChild(text);
        transcriptDiv.appendChild(entry);
        transcriptDiv.scrollTop = transcriptDiv.scrollHeight; // Auto-scroll
    });

    socket.on('processing_complete', (data) => {
        statusDiv.textContent = `Status: Idle. Session '${data.session_name}' saved.`;
        alert(`Session '${data.session_name}' has been saved successfully.`);
    });
});

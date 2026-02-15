const videoFeed = document.getElementById('videoFeed');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const signText = document.getElementById('signText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceVal = document.getElementById('confidenceVal');
const sentenceOutput = document.getElementById('sentenceOutput');

let pollingInterval = null;

function startDetection() {
    fetch('/start')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                videoFeed.src = "/video_feed";
                statusDot.classList.add('active');
                statusText.innerText = "Live Analysis";

                // Start polling for data
                if (pollingInterval) clearInterval(pollingInterval);
                pollingInterval = setInterval(updateStats, 100);
            }
        })
        .catch(err => console.error("Error starting:", err));
}

function stopDetection() {
    fetch('/stop')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                // Stop the video feed by setting src to placeholder or empty
                videoFeed.src = "";  // Or a placeholder image
                statusDot.classList.remove('active');
                statusText.innerText = "System Idle";

                if (pollingInterval) clearInterval(pollingInterval);
            }
        });
}

function resetApp() {
    fetch('/reset')
        .then(response => response.json())
        .then(data => {
            signText.innerText = "--";
            confidenceBar.style.width = "0%";
            confidenceVal.innerText = "0%";
            sentenceOutput.innerText = "";
        });
}

function updateStats() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            const pred = data.prediction;

            // Update Text
            if (pred.sign !== "--") {
                signText.innerText = pred.sign.toUpperCase();

                // Animate text color based on confidence?
                // signText.style.textShadow = `0 0 ${pred.confidence * 30}px rgba(0, 210, 255, 0.8)`;
            } else {
                signText.innerText = "--";
            }

            // Update Bar
            const confPercent = Math.round(pred.confidence * 100);
            confidenceBar.style.width = `${confPercent}%`;
            confidenceVal.innerText = `${confPercent}%`;

            // Update Sentence
            if (data.sentence) {
                sentenceOutput.innerText = data.sentence;
            } else {
                sentenceOutput.innerText = "Waiting for input...";
            }
        })
        .catch(err => console.error("Polling error:", err));
}

// Initialize with a placeholder or blank state
// videoFeed.src = "/static/placeholder.png"; 

document.addEventListener('keydown', function(event) {
    fetch('/key_press', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ key: event.key })
    })
    .then(response => response.json())
    .then(data => {
        let curText=document.getElementById('text-input').textContent;
        if (data.key=="Backspace") {
            curText=curText.substring(0, curText.length-1);
        } else if (data.valid==true) {
            curText+=data.key;
        } else if (data.key=="Enter") {
            clearInterval();
            window.location.href="/results";
        }
        document.getElementById('text-input').textContent = curText;
    })
    .catch(error => console.error('Error:', error));
});

let counter = 0;
function updateCounter() {
    counter++;
    document.getElementById('timer').textContent = "Time: "+counter+"s";
}
setInterval(updateCounter, 1000);

document.addEventListener("DOMContentLoaded", () => {
    let keystrokeData = [];
    let lastKeyTime = null;
    let lastKey = null;

    document.addEventListener("keydown", (event) => {
        let pressTime = performance.now();

        // Store the key press data
        let entry = {
            key: event.key,
            pressTime: pressTime,
            event: "press",
            flightTime: lastKeyTime ? pressTime - lastKeyTime : null
        };
        keystrokeData.push(entry);

        // Track inter-key latency
        if (lastKey) {
            keystrokeData.push({
                keyPair: `${lastKey} → ${event.key}`,
                interKeyLatency: pressTime - lastKeyTime
            });
        }

        lastKey = event.key;
        lastKeyTime = pressTime;
    });

    document.addEventListener("keyup", (event) => {
        let releaseTime = performance.now();

        // Find the corresponding press event and calculate dwell time
        for (let data of keystrokeData) {
            if (data.key === event.key && data.event === "press" && !data.dwellTime) {
                data.dwellTime = releaseTime - data.pressTime;
                break;
            }
        }
    });

    // Function to export collected data (modify as needed)
    function exportData() {
        console.log("Keystroke Data:", keystrokeData);
        // Send this data securely to your server for analysis if needed
    }

    // Example: Export data when form is submitted
    document.querySelector("form")?.addEventListener("submit", (event) => {
        event.preventDefault();
        exportData();
    });
});

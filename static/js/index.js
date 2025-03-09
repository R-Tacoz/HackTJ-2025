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
            exportData();
            // window.location.href="/results";
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
        if (lastKey) {
            keystrokeData.push({
                time: pressTime - lastKeyTime
            });
        }
        lastKey = event.key;
        lastKeyTime = pressTime;
    });

    document.addEventListener("keyup", (event) => {
        let releaseTime = performance.now();
        keystrokeData.push({
            time: releaseTime - lastKeyTime,
            key: event.key
        });
    });
});


function exportData() {
    console.log("here");
    const response = fetch("/store", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({data: keystrokeData}),
    })
    console.log("here");
    return response.json();
}
let keystrokeData = [];
let currentKeyStartTime = null;
let currentKey = null;
let lastKeyStartTime = null;
let lastKey = null;

document.addEventListener('keydown', function(event) {
    if (document.getElementById('text-input') == null)
        return;

    let key = event.key;
    let keyIsValid = /^[a-zA-Z ]$/.test(key);
    let textBoxText = document.getElementById('text-input').textContent;

    if (keyIsValid) {
        key = key.toLowerCase()
        textBoxText += key;

        let pressTime = performance.now();

        // begin key hold
        currentKey = key;
        currentKeyStartTime = pressTime;

        // end key pair elapsed
        if (lastKey) {
            keystrokeData.push({
                key_pair: lastKey + key,
                elapsed_time: pressTime - lastKeyStartTime
            });
        }

        // begin key pair elapsed
        lastKey = key;
        lastKeyStartTime = pressTime;
    }
    else if (key == "Backspace") {
        textBoxText=textBoxText.substring(0, textBoxText.length-1);
    }
    else if (key == "Enter") {
        // document.getElementById('text-input').textContent = "";
        clearInterval();
        exportData();
        window.location.href = "/results";

    }
    
    document.getElementById('text-input').textContent = textBoxText;
});

document.addEventListener("keyup", (event) => {
    let releaseTime = performance.now();
    key = event.key;

    // end key hold
    keystrokeData.push({
        held_key: currentKey,
        dwell_time: releaseTime - currentKeyStartTime
    });
});

let counter = 0;
function updateCounter() {
    if (document.getElementById('timer') == null)
        return;

    counter++;
    document.getElementById('timer').textContent = "Time: "+counter+"s";
}
setInterval(updateCounter, 1000);

function exportData() {
    const response = fetch("/store", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({data: keystrokeData}),
    })
    console.log("here");
}
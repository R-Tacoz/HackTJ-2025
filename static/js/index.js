let keystrokeData = [];
let currentKeyStartTime = null;
let currentKey = null;
let lastKeyStartTime = null;
let lastKey = null;

document.addEventListener('keydown', function(event) {

    let key = event.key;
    let keyIsValid = /^[a-zA-Z ]$/.test(key);
    let textBoxText = document.getElementById('text-input').textContent;

    if (keyIsValid) {
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
    counter++;
    document.getElementById('timer').textContent = "Time: "+counter+"s";
}
setInterval(updateCounter, 1000);

function exportData() {
    keystrokeData.push({username: })

    const response = fetch("/store", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({data: keystrokeData}),
    })
    console.log("here");
}
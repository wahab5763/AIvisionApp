// ============================================================
// LIVE AI EYE PRO
// YOLO11n ONNX + Groq Narration + Auto Start After Tap + Progress Bar
// ============================================================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let stream = null;
let running = false;
let narrationEnabled = false;
let session = null;

let questionRecognition = null;
let commandRecognition = null;
let isListeningForCommand = false;
let appStarted = false;

let latestObjects = [];
let latestNarration = "";
let lastNarrationTime = 0;
let lastSpokenText = "";

const MODEL_PATH = "models/yolo11n.onnx";
const INPUT_SIZE = 640;
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

const CONFIDENCE_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.45;

const COCO_CLASSES = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
"tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
"potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
"microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"
];

// ============================================================
// LOADING SCREEN
// ============================================================

function showLoading(message = "Loading...", percent = 10) {
    const screen = document.getElementById("loadingScreen");
    const text = document.getElementById("loadingText");
    const bar = document.getElementById("progressBar");

    if (screen) screen.style.display = "flex";
    if (text) text.innerText = message;
    if (bar) bar.style.width = percent + "%";
}

function updateLoading(message, percent) {
    const text = document.getElementById("loadingText");
    const bar = document.getElementById("progressBar");

    if (text) text.innerText = message;
    if (bar) bar.style.width = percent + "%";
}

function hideLoading() {
    const screen = document.getElementById("loadingScreen");
    if (screen) screen.style.display = "none";
}

// Tap anywhere on loading screen to start app
document.addEventListener("click", (event) => {
    const loading = document.getElementById("loadingScreen");

    if (loading && loading.style.display !== "none" && !appStarted) {
        startApp();
        return;
    }

    if (!isInteractiveElement(event.target) && appStarted) {
        startCommandListening();
    }
}, true);

// ============================================================
// STATUS HELPERS
// ============================================================

function setStatus(msg) {
    const statusEl = document.getElementById("status");
    if (statusEl) statusEl.innerText = msg;
}

function setVoiceStatus(msg) {
    const voiceEl = document.getElementById("voiceStatus");
    if (voiceEl) voiceEl.innerText = msg;
}

function setHeard(text) {
    const heardEl = document.getElementById("heardQuestion");
    if (heardEl) heardEl.innerText = text;
}

function setAnswer(text) {
    const answerEl = document.getElementById("answer");
    if (answerEl) answerEl.innerText = text;
}

function isInteractiveElement(element) {
    if (!element) return false;
    const tag = element.tagName ? element.tagName.toLowerCase() : "";
    return tag === "button" || tag === "input" || tag === "textarea" || tag === "select" || tag === "a";
}

// ============================================================
// APP START FLOW
// ============================================================

async function startApp() {
    if (appStarted) return;

    appStarted = true;
    narrationEnabled = true;

    try {
        showLoading("Starting assistant...", 10);
        speak("Starting AI Eye Assistant.", true);

        updateLoading("Requesting camera permission...", 25);
        await startCameraOnly();

        updateLoading("Camera started. Loading AI model...", 50);

        if (!session) {
            await loadModel();
        }

        updateLoading("AI model ready. Starting live detection...", 85);

        running = true;
        detectLoop();

        updateLoading("Ready. Audio narration is active.", 100);

        setTimeout(() => {
            hideLoading();
            speak("Assistant is ready. Camera and audio narration are active. Tap anywhere and say a command.", true);
        }, 700);

        setStatus("Camera and YOLO model ready.");
        setVoiceStatus("Tap anywhere and say a command.");

    } catch (err) {
        console.error(err);
        appStarted = false;
        updateLoading("Could not start. Check camera permission and refresh.", 0);
        setStatus("Startup error: " + err.name + " - " + err.message);
        speak("Could not start the assistant. Please allow camera permission and try again.", true);
    }
}

// Start camera only first, before loading model
async function startCameraOnly() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 640 },
            height: { ideal: 480 }
        },
        audio: false
    });

    video.srcObject = stream;
    video.setAttribute("playsinline", true);
    video.muted = true;

    await new Promise(resolve => {
        video.onloadedmetadata = () => resolve();
    });

    await video.play();
}

async function loadModel() {
    setStatus("Loading YOLO11n ONNX model...");

    session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"]
    });

    setStatus("YOLO11n loaded successfully.");
}

// Manual fallback button
async function startCameraAndNarration() {
    if (!appStarted) {
        await startApp();
        return;
    }

    narrationEnabled = true;

    if (!running) {
        await startCameraOnly();
        running = true;
        detectLoop();
    }

    speak("Camera and audio narration started.", true);
}

function startNarration() {
    narrationEnabled = true;
    speak("Audio narration started.", true);
    setStatus("Audio narration started.");
}

function stopAll() {
    running = false;
    narrationEnabled = false;
    speechSynthesis.cancel();

    if (questionRecognition) {
        try { questionRecognition.stop(); } catch (e) {}
    }

    if (commandRecognition) {
        try { commandRecognition.stop(); } catch (e) {}
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const descEl = document.getElementById("description");
    if (descEl) descEl.innerText = "Camera stopped.";

    const objectsEl = document.getElementById("objects");
    if (objectsEl) objectsEl.innerText = "No objects detected.";

    setStatus("Stopped.");
    setVoiceStatus("Stopped. Tap Start Camera + Audio to begin again.");
}

// ============================================================
// SPEECH OUTPUT
// ============================================================

function speak(text, force = false) {
    if ((!narrationEnabled && !force) || !text) return;

    if (speechSynthesis.speaking && !force) return;
    if (text === lastSpokenText && !force) return;

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.92;
    utter.pitch = 1.0;
    utter.volume = 1.0;

    const voices = speechSynthesis.getVoices();
    const voice =
        voices.find(v => v.lang && v.lang.toLowerCase().startsWith("en")) ||
        voices[0];

    if (voice) utter.voice = voice;

    speechSynthesis.speak(utter);
    lastSpokenText = text;
}

// ============================================================
// PREPROCESS
// ============================================================

function preprocessFrame() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;

    const tctx = tempCanvas.getContext("2d");
    tctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = tctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const input = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        input[i] = imageData[i * 4] / 255.0;
        input[i + INPUT_SIZE * INPUT_SIZE] = imageData[i * 4 + 1] / 255.0;
        input[i + 2 * INPUT_SIZE * INPUT_SIZE] = imageData[i * 4 + 2] / 255.0;
    }

    return new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// ============================================================
// YOLO POSTPROCESS
// ============================================================

function iou(boxA, boxB) {
    const x1 = Math.max(boxA.x1, boxB.x1);
    const y1 = Math.max(boxA.y1, boxB.y1);
    const x2 = Math.min(boxA.x2, boxB.x2);
    const y2 = Math.min(boxA.y2, boxB.y2);

    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    const boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
    const boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);

    return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

function nonMaxSuppression(boxes) {
    boxes.sort((a, b) => b.confidence - a.confidence);

    const selected = [];

    while (boxes.length > 0) {
        const chosen = boxes.shift();
        selected.push(chosen);

        boxes = boxes.filter(box =>
            box.classId !== chosen.classId ||
            iou(box, chosen) < IOU_THRESHOLD
        );
    }

    return selected;
}

function estimatePositionAndDistance(box) {
    const centerX = (box.x1 + box.x2) / 2;
    const boxHeight = box.y2 - box.y1;

    let position = "in front of you";

    if (centerX < video.videoWidth * 0.33) position = "on your left";
    else if (centerX > video.videoWidth * 0.66) position = "on your right";

    const ratio = boxHeight / video.videoHeight;

    let distance = "far away";
    if (ratio > 0.5) distance = "very close";
    else if (ratio > 0.3) distance = "close";
    else if (ratio > 0.18) distance = "about one to two meters away";
    else if (ratio > 0.1) distance = "a few meters away";

    return { position, distance };
}

async function runYOLO() {
    const inputTensor = preprocessFrame();

    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data;

    let boxes = [];

    for (let i = 0; i < 8400; i++) {
        const x = output[i];
        const y = output[8400 + i];
        const w = output[16800 + i];
        const h = output[25200 + i];

        let maxProb = 0;
        let classId = -1;

        for (let c = 4; c < 84; c++) {
            const prob = output[c * 8400 + i];
            if (prob > maxProb) {
                maxProb = prob;
                classId = c - 4;
            }
        }

        if (maxProb > CONFIDENCE_THRESHOLD) {
            const x1 = (x - w / 2) * video.videoWidth / INPUT_SIZE;
            const y1 = (y - h / 2) * video.videoHeight / INPUT_SIZE;
            const x2 = (x + w / 2) * video.videoWidth / INPUT_SIZE;
            const y2 = (y + h / 2) * video.videoHeight / INPUT_SIZE;

            boxes.push({
                classId,
                label: COCO_CLASSES[classId],
                confidence: maxProb,
                x1, y1, x2, y2
            });
        }
    }

    boxes = nonMaxSuppression(boxes);

    return boxes.map(box => {
        const { position, distance } = estimatePositionAndDistance(box);
        return { ...box, position, distance };
    });
}

// ============================================================
// DRAW
// ============================================================

function wrapText(text, x, y, maxWidth, lineHeight) {
    const words = text.split(" ");
    let line = "";

    for (let n = 0; n < words.length; n++) {
        const testLine = line + words[n] + " ";
        const metrics = ctx.measureText(testLine);

        if (metrics.width > maxWidth && n > 0) {
            ctx.fillText(line, x, y);
            line = words[n] + " ";
            y += lineHeight;
        } else {
            line = testLine;
        }
    }

    ctx.fillText(line, x, y);
}

function drawDetections(objects, narration) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    objects.forEach(obj => {
        ctx.strokeStyle = "#00FF88";
        ctx.lineWidth = 3;

        ctx.strokeRect(obj.x1, obj.y1, obj.x2 - obj.x1, obj.y2 - obj.y1);

        ctx.fillStyle = "#00FF88";
        ctx.font = "18px Arial";

        ctx.fillText(
            `${obj.label} ${(obj.confidence * 100).toFixed(0)}%`,
            obj.x1,
            obj.y1 > 20 ? obj.y1 - 5 : obj.y1 + 20
        );
    });

    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(0, canvas.height - 130, canvas.width, 130);

    ctx.fillStyle = "white";
    ctx.font = "22px Arial";

    wrapText(narration, 20, canvas.height - 85, canvas.width - 40, 30);
}

// ============================================================
// GROQ NARRATION
// ============================================================

async function generateNaturalNarration(objects) {
    const now = Date.now();

    if (now - lastNarrationTime < 3000 && latestNarration) {
        return latestNarration;
    }

    if (!objects || objects.length === 0) {
        latestNarration = "I do not detect any clear object right now. Move carefully.";
        lastNarrationTime = now;
        return latestNarration;
    }

    const prompt = `
You are a real-time navigation assistant for a blind person.

STRICT RULES:
- Only describe detected objects
- Mention left/right/front
- Mention distance
- Prioritize obstacles, people, vehicles, chairs
- Be precise
- No generic statements
- No assumptions
- Keep it under two short sentences

Detected objects:
${JSON.stringify(objects)}

Generate one short spoken navigation response:
`;

    try {
        const response = await fetch(GROQ_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${window.GROQ_API_KEY}`
            },
            body: JSON.stringify({
                model: "llama-3.3-70b-versatile",
                messages: [
                    { role: "system", content: "You are a blind navigation assistant." },
                    { role: "user", content: prompt }
                ],
                temperature: 0.2,
                max_tokens: 100
            })
        });

        const data = await response.json();

        latestNarration = data.choices[0].message.content.trim();
        lastNarrationTime = now;

        return latestNarration;

    } catch (err) {
        console.error(err);
        return buildLocalNarration(objects);
    }
}

function buildLocalNarration(objects) {
    if (!objects || objects.length === 0) {
        return "I do not detect any clear object right now. Move carefully.";
    }

    const top = objects.slice(0, 3);
    return top.map(o => `${o.label} ${o.position}, ${o.distance}`).join(". ") + ".";
}

// ============================================================
// MAIN LOOP
// ============================================================

async function detectLoop() {
    while (running) {
        try {
            latestObjects = await runYOLO();

            const objectsEl = document.getElementById("objects");
            if (objectsEl) {
                objectsEl.innerHTML =
                    latestObjects.length > 0
                        ? latestObjects.map(
                            o => `• ${o.label} — ${o.position}, ${o.distance}`
                        ).join("<br>")
                        : "No clear objects detected.";
            }

            const narration = await generateNaturalNarration(latestObjects);

            const descEl = document.getElementById("description");
            if (descEl) descEl.innerText = narration;

            drawDetections(latestObjects, narration);
            speak(narration);

        } catch (err) {
            console.error(err);
            setStatus("Detection error: " + err.message);
        }

        await new Promise(resolve => setTimeout(resolve, 900));
    }
}

// ============================================================
// VOICE COMMANDS + QUESTIONS
// ============================================================

function createRecognition(onResultCallback) {
    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        alert("Speech recognition is not supported. Use Chrome or Edge.");
        return null;
    }

    const rec = new SpeechRecognition();
    rec.lang = "en-US";
    rec.continuous = false;
    rec.interimResults = false;
    rec.maxAlternatives = 1;

    rec.onresult = (event) => {
        const spokenText = event.results[0][0].transcript.toLowerCase().trim();
        onResultCallback(spokenText);
    };

    rec.onerror = (event) => {
        setVoiceStatus("Voice error: " + event.error);
        speak("I could not hear clearly. Please try again.", true);
    };

    rec.onend = () => {
        isListeningForCommand = false;
    };

    return rec;
}

function startCommandListening() {
    if (isListeningForCommand) return;

    commandRecognition = createRecognition(processVoiceCommand);
    if (!commandRecognition) return;

    isListeningForCommand = true;
    setVoiceStatus("Listening for command...");
    speak("Listening. Say a command.", true);

    try {
        commandRecognition.start();
    } catch (err) {
        console.error(err);
        isListeningForCommand = false;
    }
}

async function processVoiceCommand(command) {
    setHeard("Command heard: " + command);

    if (command.includes("start camera and audio") || command.includes("start everything")) {
        await startCameraAndNarration();
        return;
    }

    if (command.includes("start audio") || command.includes("start narration")) {
        startNarration();
        return;
    }

    if (command.includes("ask question") || command.includes("question")) {
        startVoiceQuestion();
        return;
    }

    if (command.includes("describe") || command.includes("what do you see")) {
        const text = latestNarration || buildLocalNarration(latestObjects);
        setAnswer(text);
        speak(text, true);
        return;
    }

    if (command.includes("stop")) {
        stopAll();
        speak("Stopped.", true);
        return;
    }

    if (command.includes("help")) {
        const helpText =
            "You can say: start camera and audio, ask question, describe surroundings, or stop everything.";
        setAnswer(helpText);
        speak(helpText, true);
        return;
    }

    speak("I did not understand. Say help to hear commands.", true);
}

function initVoiceRecognition() {
    questionRecognition = createRecognition(processVoiceQuestion);

    if (!questionRecognition) return null;

    questionRecognition.onstart = () => {
        setVoiceStatus("Listening... Please ask your question.");
    };

    return questionRecognition;
}

function startVoiceQuestion() {
    if (!questionRecognition) {
        questionRecognition = initVoiceRecognition();
    }

    if (questionRecognition) {
        try {
            setVoiceStatus("Listening for your question.");
            speak("Ask your question now.", true);
            questionRecognition.start();
        } catch (err) {
            console.error(err);
        }
    }
}

function processVoiceQuestion(question) {
    setHeard("You asked: " + question);

    let answer = "I cannot determine that right now.";

    if (!latestObjects || latestObjects.length === 0) {
        answer = "I do not detect any clear object right now.";
    }

    else if (question.includes("how many")) {
        const labels = [...new Set(latestObjects.map(o => o.label))];

        for (const label of labels) {
            if (question.includes(label) || question.includes(label + "s")) {
                const count = latestObjects.filter(o => o.label === label).length;
                answer = `I detect ${count} ${label}${count > 1 ? "s" : ""}.`;
                break;
            }
        }
    }

    else if (
        question.includes("where is") ||
        question.includes("is there") ||
        question.includes("do you see") ||
        question.includes("can you see")
    ) {
        let found = false;

        for (const obj of latestObjects) {
            if (question.includes(obj.label) || question.includes(obj.label + "s")) {
                answer = `Yes, there is a ${obj.label} ${obj.position}, ${obj.distance}.`;
                found = true;
                break;
            }
        }

        if (!found) answer = "No, I do not detect that right now.";
    }

    else if (
        question.includes("what do you see") ||
        question.includes("describe")
    ) {
        answer = latestNarration || buildLocalNarration(latestObjects);
    }

    deliverVoiceAnswer(answer);
}

function deliverVoiceAnswer(answer) {
    setAnswer(answer);
    setVoiceStatus("Answer ready. Tap anywhere to speak another command.");
    speak(answer, true);
}
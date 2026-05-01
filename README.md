# Live AI Eye Pro Accessible

Features:
- YOLO11n ONNX Runtime Web object detection
- Groq natural narration
- Browser speech output
- Voice question answering
- Full-screen tap-anywhere voice command mode

Commands:
- start camera
- start audio
- start camera and audio
- ask question
- describe surroundings
- stop everything
- help

Setup:
1. Put yolo11n.onnx in /models, or run: python setup_model.py
2. Add Groq key in config.js
3. Run: python -m http.server 8000
4. Open: http://localhost:8000/index.html?v=3

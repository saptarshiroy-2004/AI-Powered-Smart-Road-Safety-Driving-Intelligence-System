<div id="top" align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:00F0FF&height=250&section=header&text=AI%20Smart%20Road%20Safety&fontSize=65&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Advanced%20Driving%20Intelligence%20System&descAlignY=60&descAlign=62" />

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00F0FF&center=true&vCenter=true&width=800&lines=Advanced+Driver+Assistance+System+(ADAS);Real-time+Driver+Monitoring+(DMS);Powered+by+PyTorch+%26+Computer+Vision;Smart+Telematics+%26+Edge+AI)](https://git.io/typing-svg)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB"/>
  <br>
  <img src="https://img.shields.io/badge/status-active-brightgreen.svg"/>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg"/>
</p>
</div>

---

## 🌟 Overview
**Smart Road Safety & Driving Intelligence System** is an AI-powered, Edge-deployable solution designed to significantly reduce road accidents. By combining an **Advanced Driver Assistance System (ADAS)** and a **Driver Monitoring System (DMS)**, it actively analyzes both the inward vehicle cabin environment and the outward road conditions in real-time.

<div align="center">
  <!-- Interactive Autonomous Driving Dashboard GIF -->
  <img src="https://cdn.dribbble.com/users/3281732/screenshots/8159457/media/9e7bfb83b0bd704e6e297fc9ebf52210.gif" alt="ADAS Dashboard Demo" width="90%" style="border-radius: 10px; box-shadow: 0px 4px 15px rgba(0,0,0,0.5);"/>
</div>

---

## 🚀 Key Features

### 🛣️ Advanced Driver Assistance Systems (ADAS)
- **Forward Collision Warning (FCW):** Calculates distance to the vehicle ahead and warns if a collision is imminent.
- **Lane Departure Warning (LDW):** Detects if the vehicle is drifting out of its lane.
- **Traffic Sign Recognition (TSR):** Identifies speed limits, stop signs, and traffic lights on the fly.

### 👁️ Driver Monitoring System (DMS)
- **Drowsiness Detection:** Analyzes Eye Aspect Ratio (EAR) to detect micro-sleeps and yawning.
- **Distraction Detection:** Identifies abnormal head posing, smartphone usage, or smoking.

### 📊 Telematics & Cloud Dashboard
- **Driving Safety Score:** Evaluates comprehensive driver safety behavior over time.
- **Trip Analytics Engine:** Visualizes trip paths, harsh braking events, and speed analytics via a sleek React dashboard.

---

## 🧠 System Architecture

```mermaid
graph TD;
    A[Camera Streams] --> B(Frames Pre-processing)
    B --> C{AI Inference Engine}
    C -->|YOLO/OpenCV| D[ADAS Module]
    C -->|MediaPipe/PyTorch| E[DMS Module]
    D --> F[Edge Processor - Jetson Nano]
    E --> F
    F --> G[Local Alert System]
    F -->|IoT/MQTT| H(Cloud Backend - Node.js)
    H --> I[Web Dashboard - React]
```

---

## 🛠️ Tech Stack & Hardware

### ⚙️ Hardware Recommendations
- **Inference Edge Engine:** NVIDIA Jetson Nano / Orin Nano, Raspberry Pi with Edge TPU.
- **Sensors:** 1080p Dashcam (Front), IR In-cabin Camera (Driver), GPS Module, IMU.

### 💻 Software Architecture
*   **Computer Vision Frameworks:** PyTorch, OpenCV, Ultralytics YOLO, MediaPipe, Dlib.
*   **Edge Optimization:** ONNX, TensorRT (for achieving 30+ FPS capability).
*   **Backend & Telemetry:** Node.js, Python (FastAPI), PostgreSQL, MQTT.
*   **Frontend Dashboard:** React.js, TailwindCSS.

---

## 🚦 Roadmap

- [x] **Phase 1:** Core Deep Learning Pipeline creation (State Farm Dataset).
- [ ] **Phase 2:** Lane Detection, Distance Estimation & Sensor Fusion integration.
- [ ] **Phase 3:** Edge AI optimization on NVIDIA Jetson (ONNX/TensorRT).
- [ ] **Phase 4:** Cloud Dashboard & Telemetry portal development.
- [ ] **Phase 5:** In-vehicle Real-world Testing & Tuning.

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">
</div>

## 🏁 Getting Started

### Prerequisites
Make sure you have the following installed on your machine:
* Python 3.9+ 
* CUDA-enabled GPU (optional but highly recommended for training models)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/saptarshiroy-2004/AI-Powered-Smart-Road-Safety-Driving-Intelligence-System.git
   cd AI-Powered-Smart-Road-Safety-Driving-Intelligence-System
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt  # (To be added once environment is finalized)
   ```

### Running the System
```bash
python src/engine/train.py  # To run the training pipeline
python main.py              # To start inference (once pipeline is built)
```

---

<div align="center">
  <h3>Built with ❤️ by an ambitious AI Developer.</h3>
  <p>If you find this repository helpful, consider giving it a ⭐!</p>
  
  <a href="#top">
    <img src="https://img.shields.io/badge/Back_to_Top-000?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</div>

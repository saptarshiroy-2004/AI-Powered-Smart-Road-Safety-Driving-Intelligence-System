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

### 🛣️ Advanced Driver Assistance Systems (ADAS) & Sensor Fusion
- **Multi-Modal Sensor Integration:** Real-time telemetry monitoring for Lidar, Long-Range Radar, Camera Systems, and Ultrasound.
- **Collision & Lane Warning Systems:** Evaluates trajectories to warn against imminent forward collisions and lane drifting.

### 👁️ Intelligent Driver Monitoring System (DMS)
- **Driver Attention Detection:** Calculates real-time focus ratios ($A = t_{focus} / t_{total}$) using advanced gaze tracking and facial landmark extraction CNNs.
- **Physiological Stress Monitoring:** Implements Heart Rate Variability (HRV) calculations via simulated ECG feeds to detect latent driver fatigue and drowsiness before they manifest visually.

### 🔮 Unsupervised Behavioral Clustering (K-Means)
- **Latent State Anomaly Detection:** Moves beyond standard supervised classification by utilizing an unsupervised K-Means clustering algorithm to identify hidden, novel driver behavior patterns (e.g., Micro-Sleep Risk, Distraction Anomalies) without requiring pre-labeled data.
- **Dynamic Risk Assessment:** Plots current driver telemetry against learned cluster centroids to continuously evaluate and score real-time driving risk.

### 📊 Interactive Streamlit Telemetry Dashboard
- **Live Inference Engine:** Processes dashcam frames through a custom PyTorch 3-Block CNN architecture to identify distinct driver states (based on the State Farm dataset).
- **Neural X-Ray:** Visualizes deep learning feature maps to explain how the model abstracts spatial variables and edges.

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
*   **Deep Learning & Vision:** PyTorch, OpenCV, Ultralytics YOLO, MediaPipe, Dlib.
*   **Unsupervised Learning:** Scikit-Learn (K-Means Clustering).
*   **Dashboard & Visualization:** Streamlit, Matplotlib, Pandas, Numpy.
*   **Edge Optimization:** ONNX, TensorRT (for achieving 30+ FPS capability).

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
streamlit run app.py          # To launch the Interactive Telemetry Dashboard
python src/engine/train.py    # To run the PyTorch training pipeline
```

---

<div align="center">
  <h3>Built with ❤️ by an ambitious AI Developer.</h3>
  <p>If you find this repository helpful, consider giving it a ⭐!</p>
  
  <a href="#top">
    <img src="https://img.shields.io/badge/Back_to_Top-000?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</div>

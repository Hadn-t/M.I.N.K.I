# 🤖 MINKI — Minimal Input Natural Kinetic Interpreter

MINKI is an AI-powered compact IoT device designed to bridge the communication gap for the speech and hearing impaired. It translates sign language into real-time **text** and **speech**, while also assisting the **visually impaired** by detecting objects in their surroundings using computer vision.

---

## 🎯 Problem Statement

Inaccessibility in communication restricts the personal and social lives of individuals with speech or hearing disabilities. MINKI aims to solve this by enabling intuitive, real-time, and voice-enabled sign language interpretation — giving people their voice back.

---

## 🚀 Features

- 🖐️ **Sign Language to Text & Speech**: Real-time conversion of ASL (and other sign languages) to text and spoken words.
- 🧠 **AI-Powered Gesture Recognition**: Built using CNN-based models for accurate hand gesture classification.
- 🧳 **Compact IoT Hardware**: Built on Raspberry Pi with minimal components to ensure portability.
- 🦯 **Object Detection for the Visually Impaired**: Uses computer vision to detect surroundings and speak out nearby objects.
- 🔊 **Offline & Real-time Support**: Works without the internet for edge-based instant responses.
- 📦 **Future Ready**: Designed to be converted into a single compact PCB for mass production.

---

## 💡 Tech Stack

### 🧠 AI & ML
- Python
- TensorFlow / PyTorch
- OpenCV
- MediaPipe

### 🧰 Hardware
- Raspberry Pi 4
- Pi Camera Module v2
- OLED Display (SSD1306)
- Micro Speaker + Audio Amp
- Battery Module (LiPo or USB-C)
- 3D-printed Enclosure (Optional)

### 🎛️ Tools
- Figma (for UX planning)
- RPi.GPIO / smbus (for I2C communication)
- gTTS / espeak (Text-to-Speech)

---

## 📸 Demo

https://user-demo-link-here.com

---

## 🧪 Installation

### Clone the Repository
```bash
git clone https://github.com/TeamHadnt/MINKI.git
cd MINKI

# 🧬 AgeStyle: Age-Adaptive Motion Style Transfer System for Virtual Reality

This repository provides open-source demo resources for **AgeStyle: Dynamic Age-Adaptive Motion Transfer in Virtual Reality**, including a demo platform (APK), system demonstration videos, and a core pipeline overview.  
The system integrates deep learning with VR technology to enable real-time generation and immersive visualization of motion styles across different age groups.

---

## 🧩 System Pipeline

The overall system pipeline is illustrated below:

![System Pipeline](media/pipeline.png).

---

## 📱 VR Demo Platform (APK Download)

A VR demo application is provided, which can be directly installed on VR devices.  
The demo includes features such as age prediction, motion style transfer, and multi-view interactive experiences.

The demo application (APK) can be accessed via the following link:  
[https://pan.baidu.com/s/1OaOYOCE84GhInVFHOcToeg](https://pan.baidu.com/s/1OaOYOCE84GhInVFHOcToeg)  
(extraction code: **375t**).

---

## 🎥 System Demo Video

<video src="media/demo_video.mp4" controls="controls" width="70%">
  Your browser does not support the video tag.
</video>

If the video cannot be played properly, you may download it for offline viewing:

[https://pan.baidu.com/s/1HGO9d7QvPyCwchdJFbWVJg?pwd=6a2u](https://pan.baidu.com/s/1HGO9d7QvPyCwchdJFbWVJg?pwd=6a2u)  
(extraction code: **6a2u**).

---

## 📄 Project Overview

**AgeStyle** leverages deep learning–based motion analysis and generation models to transform user motions into age-specific motion styles, including **children, young adults, middle-aged adults, and the elderly**.

---

## 🛠️ Style Transfer Model Usage

### Requirements

- Python 3.8+
- PyTorch >= 1.7.1
- scipy
- pyyaml
- numpy
- clip (OpenAI CLIP)

### Installation

Clone this repository and create environment:

```bash
cd ageStyle
conda create -n ageStyle python=3.8
conda activate ageStyle
```

First, install PyTorch >= 1.7.1 from [PyTorch](https://pytorch.org/).  
Then install the other dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start - Motion Style Transfer

Run the following command to perform motion style transfer:

```bash
cd model
python test.py --content_src data/xia_test/neutral_01_000.bvh \
               --style_src data/xia_test/childlike_01_000.bvh \
               --output_dir output
```

**Parameters:**
- `--content_src`: Input content BVH file (the motion you want to transform)
- `--style_src`: Input style BVH file (the target style)
- `--output_dir`: Output directory for the generated BVH file

---

## 📁 Project Structure

```
ageStyle/
├── media/                      # Demo videos and images
│   ├── demo_video.mp4
│   └── pipeline.png
├── model/                      # Core model code
│   ├── data/                   # Test data and normalization files
│   │   ├── xia_norms/          # Normalization parameters
│   │   └── xia_test/           # Test BVH files
│   ├── global_info/            # Skeleton configuration
│   ├── pretrained/             # Pretrained model weights
│   │   └── pth/                # Model checkpoints
│   ├── utils/                  # Utility functions
│   ├── output/                 # Generated output files
│   ├── test.py                 # Main inference script
│   ├── model.py                # Model definition
│   ├── networks.py             # Network architectures
│   ├── config.py               # Configuration
│   └── ...
├── platform/                   # VR demo platform
│   └── demo_platform.apk
└── README.md
```

---

## 📦 Dataset & Training Code

### Xia Dataset

The complete Xia motion style dataset can be downloaded from:  
[https://drive.google.com/file/d/16vKR9-OWleMuIIJ5G5iD3sqHF6MGyLEr/view](https://drive.google.com/file/d/16vKR9-OWleMuIIJ5G5iD3sqHF6MGyLEr/view)

### BFA Dataset & Training Code

🚧 **Coming Soon!**  
The BFA (Bandai-Namco Film Archive) dataset adapted for this network and the complete training code will be uploaded soon.

---

## 📚 Citation

If you use this project in your research, please cite the following paper:

**Feng Zhou, Chao Liu, Yiqing Huang, Ju Dai, Sen-Zhe X.**  
**AgeStyle: Dynamic Age-Adaptive Motion Transfer in Virtual Reality.**  
*Virtual Reality & Intelligent Hardware*, (), 1–13.

---

Thank you for your interest in **AgeStyle**. More updates will be released soon!

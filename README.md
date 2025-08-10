# PF-Florence
An official repository of paper "What Do You Really Want? - Open-Vocabulary Intention-Guided Object Detection meets Vision Language Model"
## 🫱 A Quick Overview
<p align="center"><img width="800" alt="image" src="https://github.com/nanase1025/PF-Florence/blob/main/asset/Fig1.png"></p> 
-  We propose OV-IGOD Bench, the first open-source benchmark for object detection guided by free-form language intentions, featuring 14k images and 32k diverse annotations created through our novel multi-stage pipeline.

-  We introduce PF-Florence with our Prompted Feature-wise Linear Modulation (P-FiLM) mechanism that enhances visual-language understanding through learnable queries, enabling more effective interpretation of complex intentions.

- We demonstrate through extensive experiments that our approach significantly outperforms state-of-the-art models by up to 29\% on AP metrics, establishing new benchmarks for intention-guided object detection.

## 🫱 Requirement
 Install the environment:
 ```bash
pip install transformers==4.51.3 flash_attn timm einops peft
pip install git+https://github.com/roboflow/supervision.git # If it is slow, you cloud install it from local
```
## 📕 Dataset
1. You can download the dataset from [here](https://drive.google.com/drive/folders/1ds8xeix5SB5GMexXg_EA91IyPaitJxYs?usp=drive_link).
2. The intention input paired with each bbox is under the key of `affordance`. So the final inputs to the VLM are the image and intention.
3. The ground truth is under the key of `target`, which is following the format of Florence-2 like:
```bash
<loc{x1}><loc{y1}><loc{x2}><loc{y2}>
```
with coordinates normalized and scaled to integers in the range [0, 999]
## 🫱 Usage
1. Prepare the dataset (both the imgs and jsons in ./data), you can download them from [here](https://drive.google.com/drive/folders/1ds8xeix5SB5GMexXg_EA91IyPaitJxYs?usp=drive_link).
2. Train
```bash
python train.py
```
3. Test
```bash
python test.py
```
💡**Note**: You may need to modify the path of ckpt.
## 🫱 Test other model / your model.
We provide a series of code for you to test your own model on the test set. 
You should First convert your output to the format like `./bench_others/format.json`, the output paired with each input is the value of `dino2florence_format`, like `"Lamp<loc_752><loc_440><loc_827><loc_605>"`

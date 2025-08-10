# PF-Florence
What Do You Really Want? - Open-Vocabulary Intention-Guided Object Detection meets Vision Language Model

## 🫱 Requirement
 Install the environment:
 ```bash
pip install transformers==4.51.3 flash_attn timm einops peft
pip install git+https://github.com/roboflow/supervision.git # If it is slow, you cloud install it from local
```
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

##  🚀 YOLO Model Inference Simplified

A lightweight and easy-to-use inference pipeline for YOLO models with support for encrypted weights.

---

## Project Structure
```text
├── metadata/               # Model weights
├── data/                   # Input videos/images
├── result/                 # Output videos/images
├── run_simple_inference.py # Main inference script
└── utils/
    ├── encryption.py       # Model encryption/decryption utilities
    └── pt2engine.py        # Model export to TensorRT engine
```

## Arguments Options
Argument	Description	Default Value
* **--model**	- Weights file name
* **--dummy_model** -	Path to dummy weights (for decryption)	
* **--output_dir** -	Output directory	
* **--input_dir** -	Input directory	
* **--weights_dir** -	Weights directory
* **--img_w**	- Image width after resizing
* **--img_h**	- Image height after resizing	
* **--conf** - Confidence threshold	0.75
* **--encrypted**	- Use encrypted weights	
* **--save_video** - Save processed videos

## Quick Start

1. Place your model weights in `./metadata`
2. Put input files in `./data`
3. Run the inference:
```bash
python run_simple_inference.py \
    --model my_model.pt \
    --img_w 1280 \
    --img_h 720 \
    --conf 0.85
```

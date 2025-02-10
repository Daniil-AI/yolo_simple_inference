## Simple inference for YOLO models

### Default structure
*./metadata* - weights  
*./data* - input video/images  
*./result* - output video/images  
*./run_simple_inference.py* - main code  
*./utils/encryption.py* - code for encrypting/decrypting models  
*./utils/pt2engine.py* - code for model export to engine  
##### *./run_simple_inference.py* - start script

### Arguments
*--model* - weights file name  
Default: multiclass_FHD_special_encrypted.pt  
*--dummy_model* - path to dummy weights, nessesary if using encrypted weights  
Default: "dummy_FHD.pt"  
*--output_dir* - literally  
Default: "result"  
*--input_dir* - ...  
Default: "data"  
*--weights_dir* - ...  
Default: "metadata"  
*--img_w* - image width after resizing  
Default: 1920  
*--img_h* - image height after resizing  
Default: 1088  
*--conf* - confidence threshold  
Default: 0.75  
*--encrypted* - True if using encrypted weights  
Default: True  
*--save_video* - True for saving processed videos  
Default: True  
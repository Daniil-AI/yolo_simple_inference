# Usage
# python run_simple_inference.py --model multiclass_FHD_encrypted.pt --img_w 1920 --img_h 1088 --conf 0.75 --encrypted True --save_video True
# python run_simple_inference.py --model multiclass_FHD.pt --img_w 1920 --img_h 1088 --conf 0.6 --encrypted False --save_video True

import argparse
from simple_inference import run_simple_inference
import os


def bool_(arg):
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        exit("Bool args must me True or False, like --sample_arg False")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="multiclass_FHD_special_encrypted.pt")
parser.add_argument("--dummy_model", type=str, default="dummy_FHD.pt")
parser.add_argument("--output_dir", type=str, default="result")
parser.add_argument("--input_dir", type=str, default="data")
parser.add_argument("--weights_dir", type=str, default="metadata")

parser.add_argument("--img_w", type=int, default=1920)
parser.add_argument("--img_h", type=int, default=1088)
parser.add_argument("--conf", type=float, default=0.75)
parser.add_argument("--encrypted", type=bool_, default=True)
parser.add_argument("--save_video", type=bool_, default=True)


args = parser.parse_args()

pwd: str = os.path.abspath(os.curdir)
run_simple_inference(
    model_path=os.path.join(pwd, args.weights_dir, args.model),
    dummy_model_path=os.path.join(pwd, args.weights_dir, args.dummy_model),
    start_data_dir=os.path.join(pwd, args.input_dir),
    end_data_dir=os.path.join(pwd, args.output_dir),
    conf=args.conf,
    imgsz=(args.img_h, args.img_w),
    encrypted_weight=args.encrypted,
    draw_video=args.save_video,
)

from ultralytics import YOLO
import argparse


def main(model_path):
    model = YOLO(model_path)
    model.export(format="engine")  # to engine tensorrt # format = 'onnx'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="metadata/fm2_best.pt")
    args = parser.parse_args()
    main(args.model_path)

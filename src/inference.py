import argparse
from ultralytics import YOLO

def predict(image_path: str, model_path: str, conf_threshold: float = 0.25):
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)
    # Visualize or save results
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    predict(args.image, args.weights, args.conf)

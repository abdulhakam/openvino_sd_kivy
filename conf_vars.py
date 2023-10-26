from pathlib import Path

model_id = Path("path/to/model")
diffusers_path = Path("./model/diffusers/" + model_id.stem)
ov_path = Path("./model/ov/" + model_id.stem)
output_dir = Path("./outputs/")
output_dir.mkdir(parents=True, exist_ok=True)

onnx_path = Path("./model/onnx/" + model_id.stem)
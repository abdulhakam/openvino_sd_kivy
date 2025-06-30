import os
import openvino as ov
from openvino.tools.mo import convert_model
from openvino.runtime import serialize
from config import SAFETENSORS_MODEL_PATH, OPENVINO_MODEL_PATH, IMAGE_WIDTH, IMAGE_HEIGHT

def convert_safetensors_to_openvino(safetensors_path=SAFETENSORS_MODEL_PATH, output_dir=OPENVINO_MODEL_PATH):
    """
    Converts a safetensors model to OpenVINO format.

    Args:
        safetensors_path (str): Path to the input safetensors model file.
        output_dir (str): Directory to save the converted OpenVINO model.
    """
    if not safetensors_path:
        print("Error: SAFETENSORS_MODEL_PATH is not set in config.py")
        return

    if not os.path.exists(safetensors_path):
        print(f"Error: Safetensors model not found at {safetensors_path}")
        return

    print(f"Converting model from {safetensors_path} to OpenVINO format...")

    try:
        # Use Model Optimizer to convert the model
        # The input shape might need adjustment based on the specific model
        model = convert_model(safetensors_path, compress_to_fp16=True)

        # Serialize the model to OpenVINO IR format
        serialize(model, os.path.join(output_dir, "model.xml"))
        print(f"Model successfully converted and saved to {output_dir}")

    except Exception as e:
        print(f"Error during model conversion: {e}")

if __name__ == "__main__":
    # Example usage:
    # Set SAFETENSORS_MODEL_PATH in config.py before running this script
    convert_safetensors_to_openvino()
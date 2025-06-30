import os

# Default configuration for text-to-image generation

# Model configuration
# Path to the input safetensors model file
# Example: "path/to/your/model.safetensors"
SAFETENSORS_MODEL_PATH = "../sdnext/models/Stable-diffusion/hentaidigitalart_v20.safetensors"

# Path where the converted OpenVINO model will be saved
# Example: "path/to/your/openvino_model"
OPENVINO_MODEL_PATH = "./ov_models/stable-diffusion-512"

# Image dimensions
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Output configuration
# Directory to save generated images and parameters
OUTPUT_DIR = "./outputs"

# Generation parameters
DEFAULT_PROMPT = "a photo of an astronaut riding a horse on the moon"
DEFAULT_NEGATIVE_PROMPT = "low resolution, blurry, pixelated"
DEFAULT_NUM_INFERENCE_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = -1 # Use -1 for random seed

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
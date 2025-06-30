import argparse
from model_converter import convert_safetensors_to_openvino
from image_generator import generate_image
from config import (
    SAFETENSORS_MODEL_PATH,
    DEFAULT_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED
)

def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation with OpenVINO")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert the safetensors model to OpenVINO format"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate an image using the OpenVINO model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt for image generation (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help=f"Negative prompt for image generation (default: '{DEFAULT_NEGATIVE_PROMPT}')"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of inference steps (default: {DEFAULT_NUM_INFERENCE_STEPS})"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=f"Guidance scale (default: {DEFAULT_GUIDANCE_SCALE})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (-1 for random) (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--safetensors_path",
        type=str,
        default=SAFETENSORS_MODEL_PATH,
        help=f"Path to the input safetensors model file (default: '{SAFETENSORS_MODEL_PATH}')"
    )

    args = parser.parse_args()

    if args.convert:
        convert_safetensors_to_openvino(safetensors_path=args.safetensors_path)
    elif args.generate:
        generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
    else:
        print("Please specify either --convert or --generate.")
        parser.print_help()

if __name__ == "__main__":
    main()
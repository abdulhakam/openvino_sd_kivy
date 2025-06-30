import os
import json
import time
from datetime import datetime
import numpy as np
from openvino.runtime import Core, CompiledModel
from transformers import CLIPTokenizer # Assuming CLIPTokenizer is needed for text encoding
from PIL import Image # Import PIL for image handling
from config import (
    OPENVINO_MODEL_PATH,
    OUTPUT_DIR,
    DEFAULT_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

def encode_text(tokenizer, compiled_text_encoder, prompt, negative_prompt):
    """
    Encodes the prompt and negative prompt using the CLIP tokenizer and text encoder.
    """
    # Tokenize prompts
    text_input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    ).input_ids
    uncond_input_ids = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    ).input_ids

    # Concatenate the two sets of embeddings
    prompt_embeds = np.concatenate([uncond_input_ids, text_input_ids])

    # Get text embeddings from the compiled text encoder
    text_embeddings = compiled_text_encoder([prompt_embeds])[compiled_text_encoder.output(0)]

    return text_embeddings

def generate_image(
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    seed=DEFAULT_SEED,
    model_path=OPENVINO_MODEL_PATH,
    output_dir=OUTPUT_DIR
):
    """
    Generates an image using OpenVINO Stable Diffusion.

    Args:
        prompt (str): The prompt for image generation.
        negative_prompt (str): The negative prompt.
        num_inference_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale.
        seed (int): Random seed. Use -1 for random.
        model_path (str): Path to the OpenVINO model directory.
        output_dir (str): Directory to save the output image and parameters.
    """
    if not os.path.exists(model_path):
        print(f"Error: OpenVINO model not found at {model_path}. Please run model_converter.py first.")
        return

    print("Loading OpenVINO model...")
    core = Core()
    # Assuming the model is saved as model.xml in the model_path directory
    model_xml_path = os.path.join(model_path, "model.xml")
    # Load individual model components
    tokenizer_path = os.path.join(model_path, "tokenizer.json") # Assuming tokenizer is a json file
    text_encoder_path = os.path.join(model_path, "text_encoder.xml")
    unet_path = os.path.join(model_path, "unet.xml")
    vae_encoder_path = os.path.join(model_path, "vae_encoder.xml")
    vae_decoder_path = os.path.join(model_path, "vae_decoder.xml")

    if not all(os.path.exists(p) for p in [tokenizer_path, text_encoder_path, unet_path, vae_encoder_path, vae_decoder_path]):
        print("Error: Individual model components not found. Please ensure tokenizer.json, text_encoder.xml, unet.xml, vae_encoder.xml, and vae_decoder.xml are in the model directory.")
        return

    print("Loading individual model components...")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.dirname(tokenizer_path)) # Load tokenizer from directory
    text_encoder_model = core.read_model(text_encoder_path)
    unet_model = core.read_model(unet_path)
    vae_encoder_model = core.read_model(vae_encoder_path)
    vae_decoder_model = core.read_model(vae_decoder_path)

    compiled_text_encoder = core.compile_model(text_encoder_model, "CPU")
    compiled_unet = core.compile_model(unet_model, "CPU")
    compiled_vae_encoder = core.compile_model(vae_encoder_model, "CPU")
    compiled_vae_decoder = core.compile_model(vae_decoder_model, "CPU")

    # TODO: Implement text encoding, diffusion loop, and VAE decoding
    pipe = None # Still a placeholder until the pipeline is implemented

    print("Generating image...")
    start_time = time.time()

    # Generate image
    # 1. Text encoding
    text_embeddings = encode_text(tokenizer, compiled_text_encoder, prompt, negative_prompt)

    # 2. Diffusion loop
    # Initialize latent noise
    if seed != -1:
        np.random.seed(seed)
    latents = np.random.randn(1, 4, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8).astype(np.float32)

    # Set up timesteps (basic linear schedule for now)
    timesteps = np.linspace(999, 0, num_inference_steps).astype(np.int64)

    # Get the UNet model's input and output names
    unet_input_names = [inp.any_name for inp in compiled_unet.inputs]
    unet_output_names = [out.any_name for out in compiled_unet.outputs]

    # Diffusion loop
    for t in timesteps:
        # Expand the latents for classifier-free guidance
        latent_model_input = np.concatenate([latents] * 2)
        latent_model_input = compiled_vae_encoder([latent_model_input])[compiled_vae_encoder.output(0)] # Assuming VAE encoder is needed here, might need adjustment

        # Predict the noise residual
        noise_pred = compiled_unet([latent_model_input, np.array([t], dtype=np.float32), text_embeddings])[unet_output_names[0]]

        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Update latents (basic Euler step)
        # This is a simplified step and might need a proper scheduler implementation
        alpha_prod_t = 1.0 # Placeholder, needs actual alpha_prod_t from a scheduler
        alpha_prod_t_prev = 1.0 # Placeholder, needs actual alpha_prod_t_prev from a scheduler
        beta_prod_t = 1.0 # Placeholder, needs actual beta_prod_t from a scheduler

        # Calculate the change in latent state
        dt = alpha_prod_t_prev - alpha_prod_t
        # This update rule is a simplification; a proper scheduler is required
        latents = latents - beta_prod_t * noise_pred * dt # Simplified update

    latent_model_output = latents # Final latent representation after diffusion

    # 3. VAE decoding
    # The VAE decoder expects input in a specific range, often scaled by 0.18215
    latent_model_output = latent_model_output / 0.18215

    # Decode the latent representation
    image = compiled_vae_decoder([latent_model_output])[compiled_vae_decoder.output(0)]

    # Postprocess the image: scale, clamp, and convert to PIL Image
    image = np.clip(((image + 1) / 2.0) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(image[0].transpose(1, 2, 0)) # Transpose to HWC format

    end_time = time.time()
    print(f"Image generation completed in {end_time - start_time:.2f} seconds.")

    # Save the image and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{timestamp}.png"
    params_filename = f"{timestamp}.json"
    image_path = os.path.join(output_dir, image_filename)
    params_path = os.path.join(output_dir, params_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    image.save(image_path)
    print(f"Image saved to {image_path}")

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "model_path": model_path,
        "generation_time_seconds": end_time - start_time
    }
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {params_path}")

if __name__ == "__main__":
    # Example usage:
    # Ensure the OpenVINO model is available at OPENVINO_MODEL_PATH
    # You can modify the parameters here or use the defaults from config.py
    generate_image()
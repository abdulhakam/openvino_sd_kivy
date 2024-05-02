from conf_vars import diffusers_path, ov_path, vae_path


############################ DIFFUSERS TO OPENVINO ############################
def diffusers_to_ov(
    diffusers, batch_size=1, height=512, width=512, num_images_per_prompt=1
):
    from optimum.intel.openvino import OVStableDiffusionPipeline

    print("Loading diffusers from: " + diffusers)

    ov_pipe = OVStableDiffusionPipeline.from_pretrained(
        diffusers, export=True, compile=False
    )
    print("reshaping loaded model...")
    ov_pipe.reshape(
        batch_size=batch_size,
        height=height,
        width=width,
        num_images_per_prompt=num_images_per_prompt,
    )
    ov_pipe.save_pretrained(ov_path.as_posix())
    print("done")
    del ov_pipe


########################### SAFETENSORS TO DIFFUSERS ##########################
def create_diffusers_from_safetensors(input_file):
    from diffusers import StableDiffusionPipeline

    print("loading: " + input_file)
    pipe = StableDiffusionPipeline.from_single_file(
        input_file,
        local_files_only=True,
        scheduler_type="euler-ancestral", # dpm | ddim | heun . . .
        load_safety_checker=False,
    )
    pipe.save_pretrained(diffusers_path.as_posix())
    print("saved diffusers to " + diffusers_path.as_posix())
    del pipe


########################### SAFETENSORS IN OPENVINO ###########################
def load_to_ov_pipeline(
    input_file, batch_size=1, height=512, width=512, num_images_per_prompt=1
):
    from diffusers import StableDiffusionPipeline
    sd_pipe = StableDiffusionPipeline.from_single_file(
        input_file,
        scheduler_type="euler-ancestral",
        load_safety_checker=False,
        vae=vae_path)
    sd_pipe.to('gpu')                                      ####################just playing around
    sd_pipe.save_pretrained("./cache/tempdiffuser")
    from optimum.intel.openvino import OVStableDiffusionPipeline
    print("loading from" + input_file)
    ov_pipe = OVStableDiffusionPipeline.from_pretrained("./cache/tempdiffuser",export=False,cache_dir='./cache',local_files_only=True)
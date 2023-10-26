from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.stacklayout import StackLayout
from conf_vars import ov_path, output_dir

###############################################################################
class CustomView(StackLayout):
    pass

Builder.load_file("maintest.kv")
class MainApp(MDApp):
    model_path = ov_path
    compiled_width=320
    compiled_height=512
    def t2img(self,pipeline, prompt,width,height,steps):
        if width != self.compiled_width or height != self.compiled_height:
            self.compiled_width=width
            self.compiled_height=height
            print("reshaping model...")
            pipeline.reshape(batch_size=1, width=width, height=height, num_images_per_prompt=1)
            print("compiling the model")
            pipeline.compile()
            print("ready")
            self.compiled_width=width
            self.compiled_height=height
        output_gpu_ov = pipeline(prompt, num_inference_steps=steps).images[0]
        output_gpu_ov.save(f"{output_dir/'outputs.png'}")

    from optimum.intel.openvino import OVStableDiffusionPipeline
    print("loading pipeline")
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(
        ov_path.as_posix(), compile=False, export=False,
    )
    ov_pipe.to("GPU")
    print("reshaping model...")
    ov_pipe.reshape(batch_size=1, width=compiled_width, height=compiled_height, num_images_per_prompt=1)
    print("compiling the model")
    ov_pipe.compile()
    print("ready")

    def build(self):
        view = CustomView()
        return view

def ginfer():
    MainApp().run()

ginfer()
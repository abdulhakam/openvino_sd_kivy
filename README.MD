# A Simple kivy gui running stable diffusion
This is a simple python application using KIVY GUI i wrote to play around with stable diffusion locally,
 it converts safetensors model to openvino before loading the app, kind of slow but works for me.

#### MAKE SURE TO DELETE `.blob` FILES EVERY NOW AND THEN FROM models/... FOLDER.

## RUNNING

- open to edit `conf_vars.py` file
- edit `model_id` path value to a safetensors file ( this is the only kind of file I tried).
- `python Ginfer.py`.
- wait for it.( takes a while ).
- do whatever you want.
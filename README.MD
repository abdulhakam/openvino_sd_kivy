# OpenVINO Stable Diffusion Kivy GUI

A desktop application built with Kivy for generating images using Stable Diffusion models optimized with OpenVINO.

## Features

*   Generate images using Stable Diffusion models.
*   Leverage OpenVINO for optimized model inference on Intel hardware.
*   User-friendly graphical interface built with Kivy.
*   Convert models to OpenVINO format.

## Requirements

*   Python 3.x
*   Kivy
*   OpenVINO Runtime 2024.6.0
*   Other dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd openvino_sd_kivy
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  (Optional) Convert your Stable Diffusion model to OpenVINO format using `model_converter.py`.

## Usage

1.  Run the main application:
    ```bash
    python main.py
    ```
2.  Use the Kivy interface to configure settings and generate images.

## Project Structure

```
.
├── .gitignore
├── config.py             # Configuration settings
├── image_generator.py    # Logic for image generation
├── main.py               # Main application entry point
├── maintest.kv           # Kivy GUI layout file
├── model_converter.py    # Script for converting models to OpenVINO format
├── requirements.txt      # Project dependencies
├── outputs/              # Directory for generated images
└── ov_models/            # Directory for OpenVINO optimized models
```

## License

[Specify your license here]

## Contributing

[Add contributing guidelines here]

## Acknowledgements

[Add acknowledgements here]
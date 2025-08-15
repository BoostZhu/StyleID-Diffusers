# StyleID: A Diffusers-based Implementation for Style Transfer

[![Paper](https://img.shields.io/badge/paper-CVPR'24-blue)](https://arxiv.org/abs/2404.09468)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Diffusers-yellow)](https://github.com/huggingface/diffusers)

This repository contains an unofficial PyTorch implementation of the CVPR 2024 paper **"Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer"** 

This project wraps the core concepts of StyleID into a `diffusers`-compatible pipeline. It uses a state processor and custom attention modules to precisely control the style and content inversion processes, enabling high-fidelity artistic style transfer.

## ✨ Features

- **Alternative `AttnProcessor` Implementation**: Provides a hook-free alternative to other implementations, offering a more modular and self-contained integration with `diffusers`.
- **Fully `diffusers` Compatible**: Implemented as a subclass of `StableDiffusionImg2ImgPipeline` for seamless integration with the Hugging Face ecosystem.
- **Modular Design**: Utilizes a `StyleIDState` processor and a `StyleIDAttnProcessor` for a clean, understandable, and extensible codebase.
- **Tunable Parameters**: Allows for dynamic adjustment of key parameters like content preservation strength (`gamma`) and attention temperature (`temperature`).
- **DDIM Inversion**: Implements the DDIM inversion process to accurately extract latent representations of content and style images.
- **Multi-Model Support**: Compatible with various Stable Diffusion versions, including v1.5, v2.0, and v2.1.
- **Style Pre-computation**: Features a `precompute_style` method to cache style features, accelerating the process when applying the same style to multiple content images.

## 🔧 Implementation Details: `AttnProcessor` vs. Hooks

There are two primary methods to modify the behavior of a `diffusers` model like the UNet: PyTorch hooks and custom attention processors.

1.  **Hook-Based Method**: This approach attaches a function (a hook) to a specific layer of the model (e.g., `unet.up_blocks[1].attentions[0]`). The hook intercepts the layer's input or output. It's a powerful and general-purpose PyTorch feature.

2.  **`AttnProcessor`-Based Method (This Project)**: This approach involves replacing a layer's attention processor object with a custom class that inherits from `AttnProcessor`. This is the officially recommended method by `diffusers` for modifying attention logic.

**This project uses the `AttnProcessor` method for the following reasons:**

- **Modularity & Stability**: The logic is fully encapsulated within the processor class, making the code cleaner and less dependent on the internal structure of the UNet. This can make the implementation more robust to future updates in the `diffusers` library.
- **Clarity**: It makes the modification explicit. Instead of having a hidden hook, the `unet.set_attn_processor(...)` call clearly signals that the attention mechanism is being replaced.
- **Best Practices**: It aligns with the current design philosophy of the `diffusers` library.

By providing this alternative, this repository serves as a useful case study for different engineering approaches to implementing the same research paper.

## 🎨 Showcase

Placing compelling visual examples here is crucial for demonstrating the capability of your project.

| Content | Style | Stylized Output |
| :---: | :---: | :---: |

| `examples/cnt/3.jpg` | `examples/sty/Sea_Landscape.png` | `examples/3.png` |

| `examples/cnt/2.jpg` | `examples/sty/udnie.png` | `examples/2.png` |


## 🚀 Getting Started

### 1. Environment Setup

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/BoostZhu/StyleID-Diffusers.git
cd StyleID-Diffusers
```

It is recommended to use a virtual environment. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Images

Place your content images in the `examples/content` directory and your style images in the `examples/style` directory.

### 3. Run Style Transfer

Use the `run_style_transfer.py` script to perform style transfer.

**Basic Usage:**
```bash
python run_style_transfer.py \
  --content "examples/content/your_content_image.jpg" \
  --style "examples/style/your_style_image.png" \
  --output "results/"
```
This will generate an image named `{style_name}_stylized_{content_name}.png` in the `results/` directory.

**Advanced Usage (with custom parameters):**
```bash
python run_style_transfer.py \
  --content "examples/content/your_content_image.jpg" \
  --style "examples/style/your_style_image.png" \
  --output "results/stylized_image.png" \
  --model "1.5" \
  --steps 50 \
  --gamma 0.8 \
  --temperature 1.5
```

### Command-Line Arguments

- `--content`: Path to the content image.
- `--style`: Path to the style image.
- `--output`: Path to the output image or directory.
- `--steps`: Number of DDIM sampling steps (default: `50`).
- `--gamma`: Content preservation strength. Higher values preserve more content (default: `0.75`).
- `--temperature`: Attention temperature scaling (default: `1.5`).
- `--no_adain`: Disable the initial AdaIN latent injection.
- `--no_attn`: Disable attention-based style injection.
- `--model`: Stable Diffusion model version to use (`1.5`, `2.0`, `2.1-base`, `2.1`).
- `--save_intermediates`: Save intermediate images from the inversion and generation process.

## 🔧 Code Structure

- `styleid_pipeline.py`: The core implementation, including `StyleIDPipeline`, `StyleIDState`, and `StyleIDAttnProcessor`.
- `run_style_transfer.py`: The command-line script for running style transfer.
- `requirements.txt`: Project dependencies.

##  Acknowledgements and Copyright Notice

This is an unofficial, community-driven implementation. The conceptual framework and core ideas of StyleID are the intellectual property of the original authors of the paper.

- **Original Paper:** [Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer](https://openaccess.thecvf.com/content/CVPR2024/html/Chung_Style_Injection_in_Diffusion_A_Training-free_Approach_for_Adapting_Large-scale_CVPR_2024_paper.html). All credit for the StyleID method goes to the original authors.
- **This Implementation:** The code in this repository is licensed under the MIT License (see `LICENSE` file). It is provided for academic and research purposes.
- **Libraries:** This project heavily relies on the excellent [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers).

## 📄 License 

This project is licensed under the [MIT License](LICENSE).

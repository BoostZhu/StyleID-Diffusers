import argparse
import cv2
import torch
import os
from glob import glob
from styleid_pipeline import StyleIDPipeline

def load_image(image_path):
    """Load and prepare image from path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img[:, :, ::-1]  # Convert BGR to RGB

def main():
    parser = argparse.ArgumentParser(description="StyleID: Batch style transfer using pre-computation.")
    parser.add_argument("--style", type=str, required=True, help="Path to the single style image.")
    parser.add_argument("--content_dir", type=str, required=True, help="Path to the directory containing content images.")
    parser.add_argument("--output_dir", type=str, default="results_batch", help="Directory to save the output images.")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--gamma", type=float, default=0.75, help="Content preservation strength (0-1).")
    parser.add_argument("--temperature", type=float, default=1.5, help="Attention temperature.")
    parser.add_argument("--no_adain", action="store_true", help="Disable initial latent AdaIN.")
    parser.add_argument("--no_attn", action="store_true", help="Disable attention-based style injection.")
    parser.add_argument("--model", type=str, default="1.5", 
                        choices=["1.5", "2.0", "2.1-base", "2.1"], 
                        help="Stable Diffusion model version.")
    
    args = parser.parse_args()
    
    # Map model version to model ID
    model_map = {
        "1.5": "runwayml/stable-diffusion-v1-5",
        "2.0": "stabilityai/stable-diffusion-2-base",
        "2.1-base": "stabilityai/stable-diffusion-2-1-base",
        "2.1": "stabilityai/stable-diffusion-2-1"
    }
    model_id = model_map[args.model]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the pipeline
    print(f"Loading StyleID pipeline with model {model_id}")
    pipeline = StyleIDPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

    # 1. Pre-compute style features
    print(f"Loading and pre-computing style from {args.style}")
    style_image = load_image(args.style)
    style_cache = pipeline.precompute_style(
        style_image=style_image,
        num_inference_steps=args.steps
    )
    print("Style pre-computation complete.")

    # 2. Find all content images
    image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    content_paths = []
    for ext in image_extensions:
        content_paths.extend(glob(os.path.join(args.content_dir, f"*{ext}")))
    
    if not content_paths:
        print(f"No images found in {args.content_dir}. Exiting.")
        return

    print(f"Found {len(content_paths)} content images to process.")

    # 3. Loop and transfer style
    style_name = os.path.splitext(os.path.basename(args.style))[0]
    for i, content_path in enumerate(content_paths):
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        print(f"\nProcessing [{i+1}/{len(content_paths)}]: {content_name}")

        content_image = load_image(content_path)

        output = pipeline.transfer_from_precomputed(
            content_image=content_image,
            style_cache=style_cache,
            num_inference_steps=args.steps,
            gamma=args.gamma,
            temperature=args.temperature,
            without_init_adain=args.no_adain,
            without_attn_injection=args.no_attn
        )

        # Save the output
        output_filename = f"{style_name}_stylized_{content_name}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        output.images[0].save(output_path)
        print(f"Saved result to {output_path}")

    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()

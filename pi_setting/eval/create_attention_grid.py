#!/usr/bin/env python3
"""
Create grid visualizations of VLM attention maps across layers and heads.

This script combines individual attention visualization PNGs into a single grid image,
showing how attention patterns evolve across transformer layers (rows) and attention
heads (columns).
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Create attention grid visualization")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing individual attention visualizations")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save grid visualizations")
    parser.add_argument("--timestep", type=int, required=True,
                       help="Rollout step to visualize")
    parser.add_argument("--camera", type=str, required=True,
                       help="Camera name (agentview or wrist)")
    parser.add_argument("--token_idx", type=int, required=True,
                       help="Token index to visualize")
    parser.add_argument("--layers", type=str, required=True,
                       help="Comma-separated layer indices (e.g., '0,1,2,3')")
    parser.add_argument("--heads", type=str, required=True,
                       help="Comma-separated head indices (e.g., '0,1,2,3,4,5,6,7')")
    parser.add_argument("--padding", type=int, default=5,
                       help="Padding between subplots in pixels")
    parser.add_argument("--label_font_size", type=int, default=12,
                       help="Font size for head/layer labels")
    parser.add_argument("--title_font_size", type=int, default=16,
                       help="Font size for title bar")
    return parser.parse_args()


def find_image_file(input_dir: Path, timestep: int, camera: str, layer: int, 
                   token_idx: int, head_idx: int) -> Path:
    """Find the PNG file for specific layer/head combination."""
    filename = f"timestep_{timestep:03d}_{camera}_layer{layer}_token{token_idx}_head{head_idx}.png"
    filepath = input_dir / filename
    return filepath


def extract_left_half(image: Image.Image) -> Image.Image:
    """Extract the left half of a side-by-side visualization (overlay part), excluding title bar."""
    width, height = image.size
    
    # The image structure:
    # Top: Title bar (white bar with text, ~30-40 pixels)
    # Bottom: Side-by-side (left=overlay, right=heatmap)
    
    # Detect title bar height by finding where the actual image content starts
    # The title bar is typically white/light colored at the top
    # We'll use a simple heuristic: title bar is approximately 40 pixels
    # But to be safe, let's detect it by looking for the white bar
    
    img_array = np.array(image)
    
    # Detect title bar height by finding where white bar ends
    # Search up to 250 pixels or 1/3 of image height
    title_bar_height = 100  # fallback if detection fails
    
    # Debug: sample row brightness values
    detected = False
    for y in range(min(250, height // 3)):  # Check first 250 pixels max
        row = img_array[y, :, :]
        row_mean = np.mean(row)
        
        if row_mean < 200:  # Lower threshold - actual image content (was 240, too sensitive)
            title_bar_height = y
            detected = True
            break
    
    if not detected:
        print(f"    Warning: Could not detect title bar, using fallback {title_bar_height}")
    
    # Extract bottom-left part (skip title bar, take left half)
    left_overlay = image.crop((0, title_bar_height, width // 2, height))
    return left_overlay


def create_grid_with_labels(images: dict, layers: list, heads: list, padding: int,
                           label_font_size: int, title_font_size: int,
                           timestep: int, camera: str, token_idx: int) -> Image.Image:
    """
    Create a grid layout with labels and title.
    
    Args:
        images: Dict mapping (layer, head) -> PIL.Image
        layers: List of layer indices
        heads: List of head indices
        padding: Padding between subplots
        label_font_size: Font size for labels
        title_font_size: Font size for title
        timestep: Rollout timestep
        camera: Camera name
        token_idx: Token index
    
    Returns:
        Combined grid image
    """
    # Get subplot dimensions from first image
    sample_img = list(images.values())[0]
    subplot_width, subplot_height = sample_img.size
    
    # Calculate dimensions
    n_rows = len(layers)
    n_cols = len(heads)
    
    # Margins for labels
    left_margin = 80    # Space for "Layer X" labels
    top_margin = 80     # Space for "Head X" labels and title
    title_height = 50   # Height of title bar
    
    # Total grid dimensions
    grid_width = left_margin + n_cols * subplot_width + (n_cols - 1) * padding
    grid_height = top_margin + title_height + n_rows * subplot_height + (n_rows - 1) * padding
    
    # Create white canvas
    canvas = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts (try to use a nice font, fall back to default)
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                       label_font_size)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                       title_font_size)
    except:
        label_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    title_text = f"Timestep {timestep:03d} | Camera: {camera} | Token: {token_idx} | " \
                f"Layers {layers[0]}-{layers[-1]} | Heads {heads[0]}-{heads[-1]}"
    # Get text bounding box for centering
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (grid_width - title_width) // 2
    title_y = 20
    draw.text((title_x, title_y), title_text, fill='black', font=title_font)
    
    # Draw head labels (top)
    for col_idx, head in enumerate(heads):
        x = left_margin + col_idx * (subplot_width + padding) + subplot_width // 2
        y = top_margin + title_height - 30
        text = f"Head {head}"
        # Center text
        bbox = draw.textbbox((0, 0), text, font=label_font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, y), text, fill='black', font=label_font)
    
    # Draw layer labels (left) and paste images
    for row_idx, layer in enumerate(layers):
        # Layer label
        x = 10
        y = top_margin + title_height + row_idx * (subplot_height + padding) + subplot_height // 2
        text = f"Layer {layer}"
        bbox = draw.textbbox((0, 0), text, font=label_font)
        text_height = bbox[3] - bbox[1]
        draw.text((x, y - text_height // 2), text, fill='black', font=label_font)
        
        # Paste subplot images
        for col_idx, head in enumerate(heads):
            if (layer, head) in images:
                img = images[(layer, head)]
                x = left_margin + col_idx * (subplot_width + padding)
                y = top_margin + title_height + row_idx * (subplot_height + padding)
                canvas.paste(img, (x, y))
            else:
                # Draw a gray box with "Missing" text if image not found
                x = left_margin + col_idx * (subplot_width + padding)
                y = top_margin + title_height + row_idx * (subplot_height + padding)
                draw.rectangle([x, y, x + subplot_width, y + subplot_height], 
                             fill='lightgray', outline='gray')
                text = "Missing"
                bbox = draw.textbbox((0, 0), text, font=label_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text((x + (subplot_width - text_w) // 2, 
                          y + (subplot_height - text_h) // 2),
                         text, fill='gray', font=label_font)
    
    return canvas


def main():
    args = parse_args()
    
    # Parse layer and head indices
    layers = [int(x) for x in args.layers.split(',')]
    heads = [int(x) for x in args.heads.split(',')]
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating grid for: timestep={args.timestep}, camera={args.camera}, token={args.token_idx}")
    print(f"  Layers: {layers}")
    print(f"  Heads: {heads}")
    
    # Load all images
    images = {}
    missing_count = 0
    
    for layer in layers:
        for head in heads:
            filepath = find_image_file(input_dir, args.timestep, args.camera, 
                                      layer, args.token_idx, head)
            
            if filepath.exists():
                # Load image and extract left half (overlay)
                img = Image.open(filepath)
                left_half = extract_left_half(img)
                images[(layer, head)] = left_half
            else:
                print(f"  Warning: Missing {filepath.name}")
                missing_count += 1
    
    if missing_count > 0:
        print(f"  Found {len(images)}/{len(layers) * len(heads)} images ({missing_count} missing)")
    else:
        print(f"  Found all {len(images)} images ✓")
    
    if len(images) == 0:
        print("  ERROR: No images found!")
        sys.exit(1)
    
    # Create grid
    print("  Creating grid...")
    grid_image = create_grid_with_labels(
        images, layers, heads, args.padding,
        args.label_font_size, args.title_font_size,
        args.timestep, args.camera, args.token_idx
    )
    
    # Save output
    output_filename = (f"timestep_{args.timestep:03d}_{args.camera}_"
                      f"token{args.token_idx}_"
                      f"layer{layers[0]}to{layers[-1]}_"
                      f"head{heads[0]}to{heads[-1]}.png")
    output_path = output_dir / output_filename
    
    grid_image.save(output_path, dpi=(300, 300))
    print(f"  Saved: {output_path}")
    print(f"  Dimensions: {grid_image.size[0]} × {grid_image.size[1]} pixels")


if __name__ == "__main__":
    main()

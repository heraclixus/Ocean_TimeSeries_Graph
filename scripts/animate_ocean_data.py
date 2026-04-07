import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os
import argparse
import sys
try:
    from skimage import exposure
except ImportError:
    print("Warning: scikit-image not installed. Histogram equalization will not be available.")
    print("Install with: pip install scikit-image")
    
    # Fallback implementation
    def exposure():
        class ExposureFallback:
            @staticmethod
            def rescale_intensity(image, in_range):
                # Simple rescaling implementation
                min_val, max_val = in_range
                return np.clip((image - min_val) / (max_val - min_val), 0, 1)
        return ExposureFallback()
    exposure = exposure()

def create_ocean_animation(anom_data, mask=None, region_coords=None, fps=10, 
                          cmap='viridis', output_path='ocean_animation.mp4',
                          title_prefix='Ocean Data', vmin=None, vmax=None,
                          dpi=150, figsize=(12, 8), show_grid=False, 
                          enhance_contrast=False, center_colormap=False):
    """
    Create an animation of ocean data with highlighted region of interest.
    
    Args:
        anom_data (np.ndarray): Data array of shape (time_steps, height, width)
        mask (np.ndarray, optional): Binary mask array of shape (height, width)
        region_coords (tuple, optional): (min_y, max_y, min_x, max_x) for rectangular region
        fps (int): Frames per second in the output animation
        cmap (str): Colormap to use
        output_path (str): Path to save the output animation
        title_prefix (str): Prefix for the title of each frame
        vmin, vmax (float, optional): Color scale limits
        dpi (int): DPI for the output animation
        figsize (tuple): Figure size (width, height) in inches
        show_grid (bool): Whether to show grid lines
        enhance_contrast (bool): Whether to enhance contrast using percentile-based limits
        center_colormap (bool): Whether to center colormap around zero
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get global min/max if not provided
    if vmin is None or vmax is None:
        if enhance_contrast:
            # Use percentiles for better contrast
            flat_data = anom_data.flatten()
            vmin = np.nanpercentile(flat_data, 2) if vmin is None else vmin
            vmax = np.nanpercentile(flat_data, 98) if vmax is None else vmax
            print(f"Enhanced contrast limits: vmin={vmin:.3f}, vmax={vmax:.3f}")
        else:
            vmin = np.nanmin(anom_data) if vmin is None else vmin
            vmax = np.nanmax(anom_data) if vmax is None else vmax
    
    # Center colormap around zero if requested
    if center_colormap and vmin is not None and vmax is not None:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        print(f"Centered colormap around zero: vmin={vmin:.3f}, vmax={vmax:.3f}")
    
    # Choose colormap
    if center_colormap and 'div' not in cmap:
        # Use a diverging colormap for centered data
        cmap = 'RdBu_r'  # Red-Blue diverging colormap, reversed
    
    # Initial frame
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(anom_data[0], cmap=cmap, norm=norm, interpolation='none')
    plt.colorbar(im, ax=ax, label='Value')
    
    # Add some grid lines if requested
    if show_grid:
        # Add grid lines every 10 pixels
        ax.set_xticks(np.arange(-.5, anom_data.shape[2], 10))
        ax.set_yticks(np.arange(-.5, anom_data.shape[1], 10))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Title with frame number
    title = ax.set_title(f'{title_prefix} - Frame 0/{anom_data.shape[0]-1}')
    
    # If region coordinates are provided, draw a rectangle (preferred method)
    if region_coords is not None:
        min_y, max_y, min_x, max_x = region_coords
        width = max_x - min_x
        height = max_y - min_y
        rect = Rectangle((min_x, min_y), width, height, 
                         edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    # Otherwise, if mask is provided, create a rectangle from mask bounds
    elif mask is not None:
        # Get bounds from mask
        min_y, max_y, min_x, max_x = get_region_from_mask(mask)
        width = max_x - min_x
        height = max_y - min_y
        # Draw a rectangle based on mask bounds rather than all points
        rect = Rectangle((min_x, min_y), width, height, 
                         edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    
    # Update function for animation
    def update(frame):
        im.set_array(anom_data[frame])
        title.set_text(f'{title_prefix} - Frame {frame}/{anom_data.shape[0]-1}')
        return im, title
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=anom_data.shape[0], 
                                 blit=True, interval=1000/fps)
    
    # Save the animation
    print(f"Saving animation to {output_path}...")
    ani.save(output_path, dpi=dpi)
    plt.close()
    print(f"Animation saved successfully!")
    
    return output_path

def get_region_from_mask(mask):
    """
    Extract rectangular region bounds from a binary mask
    
    Args:
        mask (np.ndarray): Binary mask array of shape (height, width)
        
    Returns:
        tuple: (min_y, max_y, min_x, max_x) for rectangular region
    """
    # Find all True positions
    rows, cols = np.where(mask)
    
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Empty mask provided, cannot extract region bounds")
    
    # Find the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Return as min_y, max_y, min_x, max_x
    return min_row, max_row+1, min_col, max_col+1  # +1 for inclusive bounds

def create_square_mask(height, width, min_y, max_y, min_x, max_x):
    """
    Create a square mask given coordinates
    """
    mask = np.zeros((height, width), dtype=bool)
    mask[min_y:max_y, min_x:max_x] = True
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an animation of ocean data')
    parser.add_argument('--data_file', type=str, help='Path to the data file (.npy)')
    parser.add_argument('--mask_file', type=str, help='Path to the mask file (.npy)')
    parser.add_argument('--region', type=int, nargs=4, 
                       help='Region coordinates: min_y max_y min_x max_x')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--output', type=str, default='ocean_animation.mp4', 
                       help='Output animation file path')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap')
    parser.add_argument('--title', type=str, default='Ocean Data', help='Title prefix')
    parser.add_argument('--vmin', type=float, help='Minimum value for color scale')
    parser.add_argument('--vmax', type=float, help='Maximum value for color scale')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output video')
    parser.add_argument('--skip', type=int, default=1, 
                       help='Skip every N frames to reduce animation size')
    parser.add_argument('--show_grid', action='store_true', help='Whether to show grid lines')
    parser.add_argument('--enhance_contrast', action='store_true', help='Whether to enhance contrast')
    parser.add_argument('--center_colormap', action='store_true', help='Whether to center colormap')
    parser.add_argument('--histogram_equalization', action='store_true', 
                       help='Apply histogram equalization for better contrast')
    parser.add_argument('--list_colormaps', action='store_true', 
                       help='List available colormaps with descriptions')
    parser.add_argument('--suggest_colormap', action='store_true',
                       help='Suggest appropriate colormap based on data')
    
    args = parser.parse_args()
    
    # List colormaps if requested
    if args.list_colormaps:
        list_colormaps()
        sys.exit(0)
    
    # Check if running example
    if len(sys.argv) == 1 or args.data_file is None:
        print("No arguments provided. Running example...")
        run_example()
        sys.exit(0)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    anom_data = np.load(args.data_file)
    
    # Skip frames if requested
    if args.skip > 1:
        anom_data = anom_data[::args.skip]
        print(f"Skipping every {args.skip} frames. New animation length: {anom_data.shape[0]}")
    
    # Get mask or region
    mask = None
    region_coords = None
    
    if args.mask_file:
        print(f"Loading mask from {args.mask_file}...")
        mask = np.load(args.mask_file)
        
        # Get region coordinates from mask if region not specified
        if args.region is None:
            region_coords = get_region_from_mask(mask)
            print(f"Extracted region from mask: {region_coords}")
    
    if args.region:
        region_coords = args.region
        print(f"Using provided region coordinates: {region_coords}")
        
        # Create mask from region if mask not provided
        if mask is None:
            mask = create_square_mask(anom_data.shape[1], anom_data.shape[2], *region_coords)
            print(f"Created mask from region coordinates")
    
    # Suggest colormap if requested
    cmap = args.cmap
    center = args.center_colormap
    if args.suggest_colormap:
        suggested_cmap, should_center = suggest_colormap(anom_data)
        cmap = suggested_cmap
        center = should_center
        print(f"Suggested colormap: {cmap} (centered: {center})")
    
    # Apply histogram equalization if requested
    if args.histogram_equalization:
        print("Applying histogram equalization for enhanced contrast...")
        anom_data = enhance_contrast_with_histogram_equalization(anom_data)
        print("Histogram equalization applied")
    
    # Create animation
    output_path = create_ocean_animation(
        anom_data, 
        mask=mask,
        region_coords=region_coords,
        fps=args.fps,
        cmap=cmap,
        output_path=args.output,
        title_prefix=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
        show_grid=args.show_grid,
        enhance_contrast=args.enhance_contrast,
        center_colormap=center
    )
    
    print(f"Animation saved to {output_path}")

def create_example_data():
    """
    Create example data for testing the animation
    
    Returns:
        tuple: (data, region_coords) where data is of shape (100, 180, 45) and
              region_coords is (min_y, max_y, min_x, max_x)
    """
    # Create example data (100 frames, 180x45 grid)
    print("Creating example data...")
    np.random.seed(42)
    time_steps = 100
    height, width = 180, 45
    
    # Initialize with random data
    data = np.random.randn(time_steps, height, width) * 0.5
    
    # Add some temporal evolution patterns
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 25, height)
    X, Y = np.meshgrid(x, y)
    
    for t in range(time_steps):
        data[t] += 2 * np.sin(X + 0.1*t) * np.cos(Y + 0.05*t)
    
    # Define region of interest
    min_y, max_y = 60, 120
    min_x, max_x = 15, 30
    
    print(f"Example data shape: {data.shape}")
    print(f"Example region: ({min_y}, {max_y}, {min_x}, {max_x})")
    
    return data, (min_y, max_y, min_x, max_x)

def run_example():
    """Run a demonstration with various color contrast options"""
    print("Running example animations with different color contrast options...")
    
    # Create example data
    data, region_coords = create_example_data()
    
    # 1. Default settings
    output_path1 = create_ocean_animation(
        data,
        region_coords=region_coords,
        output_path='example_default.mp4',
        fps=10,
        title_prefix='Default Settings'
    )
    
    # 2. Enhanced contrast with percentiles
    output_path2 = create_ocean_animation(
        data,
        region_coords=region_coords,
        output_path='example_enhanced_contrast.mp4',
        fps=10,
        title_prefix='Enhanced Contrast',
        enhance_contrast=True
    )
    
    # 3. Centered colormap around zero
    output_path3 = create_ocean_animation(
        data,
        region_coords=region_coords,
        output_path='example_centered.mp4',
        fps=10,
        title_prefix='Centered Around Zero',
        cmap='RdBu_r',
        center_colormap=True
    )
    
    # 4. High contrast colormap
    output_path4 = create_ocean_animation(
        data,
        region_coords=region_coords,
        output_path='example_high_contrast.mp4',
        fps=10,
        title_prefix='High Contrast Colormap',
        cmap='inferno',
        enhance_contrast=True
    )
    
    # Try histogram equalization if available
    try:
        # Apply histogram equalization
        enhanced_data = enhance_contrast_with_histogram_equalization(data)
        
        # 5. Histogram equalized
        output_path5 = create_ocean_animation(
            enhanced_data,
            region_coords=region_coords,
            output_path='example_histogram_eq.mp4',
            fps=10,
            title_prefix='Histogram Equalization',
            cmap='viridis'
        )
        paths = [output_path1, output_path2, output_path3, output_path4, output_path5]
    except Exception as e:
        print(f"Histogram equalization failed: {e}")
        paths = [output_path1, output_path2, output_path3, output_path4]
    
    print("\nCreated example animations with different color contrast options:")
    for i, path in enumerate(paths, 1):
        print(f"{i}. {path}")
    print("\nTry each one to find which visualization works best for your data!")
    
    return paths

def get_available_colormaps():
    """
    Returns a list of recommended colormaps for ocean data visualization
    with descriptions
    """
    cmaps = {
        # Sequential colormaps (good for continuous data)
        'viridis': 'Default - perceptually uniform green-blue-purple',
        'plasma': 'Yellow-red-purple, good for temperature',
        'inferno': 'Yellow-red-black, high contrast',
        'cividis': 'Blue-yellow, color-vision deficiency friendly',
        'magma': 'Purple-red-yellow, high contrast',
        'turbo': 'Blue-green-yellow-red, high contrast',
        
        # Ocean-specific colormaps
        'ocean': 'Green-blue ocean colormap',
        'deep': 'Deep purple-blue colormap',
        'dense': 'Blue-green-yellow dense water colormap',
        
        # Diverging colormaps (good for anomalies, centered around zero)
        'RdBu_r': 'Red-White-Blue (reversed), good for temperature anomalies',
        'coolwarm': 'Blue-white-red, good for anomalies',
        'seismic': 'Blue-white-red, higher contrast',
        'BrBG': 'Brown-White-Blue-Green, good for precipitation anomalies',
        'PiYG': 'Pink-White-Green, good visibility',
        
        # High contrast options
        'jet': 'Classic rainbow colormap, high contrast but not perceptually uniform',
        'nipy_spectral': 'Improved rainbow colormap with better contrast',
        'gist_rainbow': 'Rainbow colormap for high contrast',
    }
    
    return cmaps

def suggest_colormap(data_array):
    """
    Suggest an appropriate colormap based on data characteristics
    """
    # Check if data has both positive and negative values
    has_positive = np.any(data_array > 0)
    has_negative = np.any(data_array < 0)
    
    if has_positive and has_negative:
        # Data has both positive and negative values - use diverging colormap
        return 'RdBu_r', True  # Colormap, center_around_zero
    else:
        # Data is all positive or all negative - use sequential colormap
        return 'viridis', False

def list_colormaps():
    """Print available colormaps with descriptions"""
    cmaps = get_available_colormaps()
    print("\nAvailable colormaps for ocean data:")
    print("-" * 70)
    for cmap, desc in cmaps.items():
        print(f"- {cmap:<15} : {desc}")
    print("-" * 70)
    print("Usage: --cmap <colormap_name>")

def enhance_contrast_with_histogram_equalization(data):
    """
    Enhance contrast using histogram equalization
    
    Args:
        data: Input data array of shape (time, height, width)
        
    Returns:
        Enhanced data array with same shape
    """
    # Create output array
    enhanced = np.zeros_like(data)
    
    # Process each frame
    for i in range(data.shape[0]):
        # Get min/max for this frame
        frame = data[i]
        p2, p98 = np.nanpercentile(frame, (2, 98))
        frame_rescaled = exposure.rescale_intensity(frame, in_range=(p2, p98))
        enhanced[i] = frame_rescaled
        
    return enhanced 
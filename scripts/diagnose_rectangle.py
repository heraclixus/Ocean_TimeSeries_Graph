import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# Create a mask
height, width = 180, 45
mask = np.zeros((height, width), dtype=bool)

# Make a rectangular region
min_y, max_y = 60, 120
min_x, max_x = 15, 30
mask[min_y:max_y, min_x:max_x] = True

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Comparing Methods for Highlighting a Rectangle Region", fontsize=16)

# 1. Original approach (many red lines)
ax1 = axes[0, 0]
ax1.set_title("Problem: Using Polygon with All Mask Points")
ax1.imshow(np.zeros((height, width)), cmap='viridis')

# This is what was causing your issue - creating a polygon from all mask points
y, x = np.where(mask.T)  # Note the transpose!
xy = np.column_stack([x, y])
poly = Polygon(xy, facecolor='none', edgecolor='red', linewidth=1)
ax1.add_patch(poly)
ax1.set_xlim(0, width)
ax1.set_ylim(height, 0)
ax1.text(5, 20, "Issue: Creates a polygon using all points\nin the mask, resulting in many red lines", 
        color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

# 2. Fixed approach - using Rectangle
ax2 = axes[0, 1]
ax2.set_title("Solution: Using Rectangle")
ax2.imshow(np.zeros((height, width)), cmap='viridis')

# Using a Rectangle - clean and clear
rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                edgecolor='red', facecolor='none', linewidth=2)
ax2.add_patch(rect)
ax2.set_xlim(0, width)
ax2.set_ylim(height, 0)
ax2.text(5, 20, "Better: Uses a Rectangle patch\nwith just the corner coordinates", 
        color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

# 3. Original mask visualization
ax3 = axes[1, 0]
ax3.set_title("The Original Mask")
ax3.imshow(mask, cmap='gray')
ax3.set_xlim(0, width)
ax3.set_ylim(height, 0)
ax3.text(5, 20, f"Mask shape: {mask.shape}\nTrue values: {np.sum(mask)}", 
        color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

# 4. Mask with proper contour
ax4 = axes[1, 1]
ax4.set_title("Mask with Proper Contour")
ax4.imshow(mask, cmap='gray')

# Draw contour around the outside only
from matplotlib import patches
contour_vertices = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
contour = patches.Polygon(contour_vertices, closed=True, 
                         fill=False, edgecolor='red', linewidth=2)
ax4.add_patch(contour)
ax4.set_xlim(0, width)
ax4.set_ylim(height, 0)
ax4.text(5, 20, "Another solution: Draw contour only\naround the outer vertices of the mask", 
        color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

plt.tight_layout()
plt.savefig('rectangle_diagnosis.png', dpi=150)
plt.show()

print("Diagnosis image saved as rectangle_diagnosis.png")
print("\nExplanation:")
print("-" * 80)
print("The issue with seeing many red lines instead of a rectangle happens because")
print("the original code was creating a polygon using ALL points in the mask, not")
print("just the corner points needed for a rectangle.")
print("\nThe best solution is to use the Rectangle patch, which just needs:")
print(f"- Starting coordinates: ({min_x}, {min_y})")
print(f"- Width: {max_x - min_x}")
print(f"- Height: {max_y - min_y}")
print("-" * 80) 
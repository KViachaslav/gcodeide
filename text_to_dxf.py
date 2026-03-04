#!/usr/bin/env python3
"""
Text to DXF Font Outline Generator

This script takes a TrueType font file, a string of text, and generates a DXF file
containing the vector outlines of the text characters, including support for holes
in characters like 'A', 'B', 'O', etc.

Requirements:
    pip install fonttools ezdxf

Usage:
    python text_to_dxf.py "Hello World" font.ttf output.dxf
"""

import sys
import argparse
import platform
import os
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
import ezdxf
from ezdxf.math import Vec2
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_system_font_paths():
    """Get standard font directories for the current operating system."""
    system = platform.system().lower()
    font_paths = []
    
    if system == "windows":
        # Windows font directories
        font_paths.extend([
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts'),
        ])
    elif system == "darwin":  # macOS
        # macOS font directories
        font_paths.extend([
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts'),
        ])
    else:  # Linux and other Unix-like systems
        # Linux font directories
        font_paths.extend([
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts'),
        ])
    
    # Filter out non-existent directories
    return [path for path in font_paths if os.path.exists(path)]


def find_all_fonts():
    """Find all TrueType and OpenType fonts on the system."""
    fonts = {}  # Dictionary to store font_name: font_path
    font_extensions = {'.ttf', '.otf', '.ttc'}
    
    for font_dir in get_system_font_paths():
        try:
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in font_extensions):
                        font_path = os.path.join(root, file)
                        try:
                            # Try to get the font name from the font file
                            font = TTFont(font_path)
                            name_table = font.get('name')
                            if name_table:
                                # Try to get the font family name (ID 1) and full name (ID 4)
                                family_name = None
                                full_name = None
                                
                                for record in name_table.names:
                                    if record.nameID == 1 and record.platformID == 3:  # Family name, Windows platform
                                        try:
                                            family_name = record.toUnicode()
                                        except:
                                            pass
                                    elif record.nameID == 4 and record.platformID == 3:  # Full name, Windows platform
                                        try:
                                            full_name = record.toUnicode()
                                        except:
                                            pass
                                
                                # Use full name if available, otherwise family name, otherwise filename
                                display_name = full_name or family_name or os.path.splitext(file)[0]
                                fonts[display_name] = font_path
                            else:
                                # Fallback to filename if no name table
                                fonts[os.path.splitext(file)[0]] = font_path
                            
                            font.close()
                        except Exception:
                            # If we can't read the font, use the filename
                            fonts[os.path.splitext(file)[0]] = font_path
        except (PermissionError, OSError):
            # Skip directories we can't access
            continue
    
    return fonts


def list_fonts():
    """List all available fonts on the system."""
    print("Searching for fonts...")
    fonts = find_all_fonts()
    
    if not fonts:
        print("No fonts found on the system.")
        return
    
    print(f"\nFound {len(fonts)} fonts:\n")
    
    # Sort fonts alphabetically
    for font_name in sorted(fonts.keys()):
        font_path = fonts[font_name]
        print(f"  {font_name}")
        print(f"    Path: {font_path}")
        print()

def list_common_fonts(args):
    """List common Adobe, Ubuntu, and Microsoft fonts found on the system."""
    common_font_names = [
        # Adobe
        "Adobe Garamond", "Myriad Pro", "Minion Pro", "Source Sans Pro", "Source Serif Pro", "Source Code Pro",
        # Microsoft
        "Arial", "Times New Roman", "Courier New", "Verdana", "Tahoma", "Georgia", "Impact", "Trebuchet MS",
        "Comic Sans MS", "Calibri", "Cambria", "Consolas", "Corbel", "Candara", "Segoe UI",
        # Ubuntu
        "Ubuntu", "Ubuntu Mono", "Ubuntu Condensed", "Ubuntu Light",
        # Common cross-platform (often included with OS or popular software)
        "Helvetica", "sans-serif", "serif", "monospace", "DejaVu Sans", "Liberation Sans", "Noto Sans"
    ]
    
    print("Searching for common fonts...")
    all_fonts = find_all_fonts()
    
    found_common_fonts = {}
    for font_display_name, font_path in all_fonts.items():
        if font_display_name.lower().startswith("noto"): # Exclude Noto fonts
            continue
        for common_name in common_font_names:
            # Check if the font_display_name starts with the common_name or contains it with a space/hyphen
            if font_display_name.lower().startswith(common_name.lower()) or \
               f" {common_name.lower()}" in font_display_name.lower() or \
               f"-{common_name.lower()}" in font_display_name.lower():
                found_common_fonts[font_display_name] = font_path
                break # Found a match for this font_display_name, move to next available font
    
    if not found_common_fonts:
        print("No common fonts found on the system.")
        return
    
    print(f"\nFound {len(found_common_fonts)} common fonts:\n")
    
    for font_name in sorted(found_common_fonts.keys()):
        font_path = found_common_fonts[font_name]
        print(f"  {font_name}")
        if args.verbose:
            print(f"    Path: {font_path}")
            print()


class DXFPen(BasePen):
    """A pen that converts font outlines to DXF polylines."""
    
    def __init__(self, msp, x_offset=0, y_offset=0, scale=1.0, curve_quality=0.5):
        super().__init__(glyphSet=None)
        self.msp = msp  # DXF model space
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.scale = scale
        self.current_path = []
        self.paths = []  # Store all paths for this glyph
        self.curve_quality = curve_quality
        
    def _transform_point(self, pt):
        """Transform point coordinates."""
        x, y = pt
        return (
            (x * self.scale) + self.x_offset,
            (y * self.scale) + self.y_offset
        )
    
    def moveTo(self, pt):
        """Start a new path."""
        if self.current_path:
            self.paths.append(self.current_path)
        self.current_path = [self._transform_point(pt)]
    
    def lineTo(self, pt):
        """Add a line to the current path."""
        self.current_path.append(self._transform_point(pt))
    
    def curveTo(self, *points):
        """Add a cubic Bezier curve to the current path."""
        if not self.current_path:
            return
            
        start = self.current_path[-1]
        cp1, cp2, end = [self._transform_point(pt) for pt in points]
        
        # Calculate curve length to determine appropriate number of segments
        # Rough approximation: sum of control polygon lengths
        dist1 = ((cp1[0] - start[0])**2 + (cp1[1] - start[1])**2)**0.5
        dist2 = ((cp2[0] - cp1[0])**2 + (cp2[1] - cp1[1])**2)**0.5
        dist3 = ((end[0] - cp2[0])**2 + (end[1] - cp2[1])**2)**0.5
        approx_length = dist1 + dist2 + dist3
        
        # Adaptive number of steps based on curve length and scale
        # More segments for longer curves and larger scales
        min_steps = 8
        max_steps = 50
        steps = max(min_steps, min(max_steps, int(approx_length * self.curve_quality)))
        
        # Generate cubic Bezier curve points
        for i in range(1, steps + 1):
            t = i / steps
            # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            
            x = mt3 * start[0] + 3 * mt2 * t * cp1[0] + 3 * mt * t2 * cp2[0] + t3 * end[0]
            y = mt3 * start[1] + 3 * mt2 * t * cp1[1] + 3 * mt * t2 * cp2[1] + t3 * end[1]
            self.current_path.append((x, y))
    
    def qCurveTo(self, *points):
        """Add a quadratic Bezier curve to the current path."""
        if not self.current_path:
            return
            
        start = self.current_path[-1]
        
        # Handle quadratic curves (can have multiple control points)
        current_start = start
        for i in range(0, len(points)):
            if i == len(points) - 1:
                # Last point is the end point
                end = self._transform_point(points[i])
                
                # If we only have one point total, it's a line
                if len(points) == 1:
                    self.current_path.append(end)
                else:
                    # Use the previous point as control point
                    if i > 0:
                        cp = self._transform_point(points[i-1])
                    else:
                        # No control point, draw line
                        self.current_path.append(end)
                        break
                    
                    # Calculate curve length for adaptive segmentation
                    dist1 = ((cp[0] - current_start[0])**2 + (cp[1] - current_start[1])**2)**0.5
                    dist2 = ((end[0] - cp[0])**2 + (end[1] - cp[1])**2)**0.5
                    approx_length = dist1 + dist2
                    
                    # Adaptive steps
                    min_steps = 6
                    max_steps = 30
                    steps = max(min_steps, min(max_steps, int(approx_length * self.curve_quality)))
                    
                    # Generate quadratic Bezier curve points
                    for j in range(1, steps + 1):
                        t = j / steps
                        # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                        mt = 1 - t
                        mt2 = mt * mt
                        t2 = t * t
                        
                        x = mt2 * current_start[0] + 2 * mt * t * cp[0] + t2 * end[0]
                        y = mt2 * current_start[1] + 2 * mt * t * cp[1] + t2 * end[1]
                        self.current_path.append((x, y))
                    
                current_start = end
            else:
                # Handle multiple control points in TrueType quadratic curves
                if i < len(points) - 1:
                    cp1 = self._transform_point(points[i])
                    
                    # Check if next point exists and is not the last
                    if i + 1 < len(points) - 1:
                        cp2 = self._transform_point(points[i + 1])
                        # Calculate implied on-curve point between control points
                        implied_end = ((cp1[0] + cp2[0]) / 2, (cp1[1] + cp2[1]) / 2)
                        
                        # Draw curve to implied point
                        dist1 = ((cp1[0] - current_start[0])**2 + (cp1[1] - current_start[1])**2)**0.5
                        dist2 = ((implied_end[0] - cp1[0])**2 + (implied_end[1] - cp1[1])**2)**0.5
                        approx_length = dist1 + dist2
                        
                        min_steps = 4
                        max_steps = 20
                        steps = max(min_steps, min(max_steps, int(approx_length * self.curve_quality)))
                        
                        for j in range(1, steps + 1):
                            t = j / steps
                            mt = 1 - t
                            mt2 = mt * mt
                            t2 = t * t
                            
                            x = mt2 * current_start[0] + 2 * mt * t * cp1[0] + t2 * implied_end[0]
                            y = mt2 * current_start[1] + 2 * mt * t * cp1[1] + t2 * implied_end[1]
                            self.current_path.append((x, y))
                        
                        current_start = implied_end
    
    def closePath(self):
        """Close the current path."""
        if self.current_path and len(self.current_path) > 1:
            # Close the path by connecting to the first point
            if self.current_path[0] != self.current_path[-1]:
                self.current_path.append(self.current_path[0])
    
    def endPath(self):
        """End the current path and add it to the paths list."""
        if self.current_path:
            self.paths.append(self.current_path)
            self.current_path = []
    
    def draw_to_dxf(self):
        """Draw all paths to the DXF model space."""
        # Make sure to add the last path
        if self.current_path:
            self.paths.append(self.current_path)
        
        for path in self.paths:
            if len(path) > 1:
                # Create a polyline for each path
                polyline = self.msp.add_lwpolyline(path, close=False)
                polyline.dxf.layer = "TEXT_OUTLINES"


def find_font_by_name(font_name):
    """Find a font file by name (case-insensitive partial matching)."""
    fonts = find_all_fonts()
    font_name_lower = font_name.lower()
    
    # First, try exact match (case-insensitive)
    for name, path in fonts.items():
        if name.lower() == font_name_lower:
            return path
    
    # Then try partial match
    matches = []
    for name, path in fonts.items():
        if font_name_lower in name.lower():
            matches.append((name, path))
    
    if len(matches) == 1:
        return matches[0][1]
    elif len(matches) > 1:
        print(f"Multiple fonts found matching '{font_name}':")
        for name, path in matches:
            print(f"  {name}")
        print("\nPlease be more specific or use the full font name.")
        return None
    else:
        print(f"No font found matching '{font_name}'.")
        print("Use --list-fonts to see available fonts.")
        return None


def _load_font_and_get_scale(font_path, font_size, font_index, verbose=False):
    if verbose:
        print(f"Loading font: {font_path}")
    try:
        font = TTFont(font_path, fontNumber=font_index)
        if verbose:
            print("Font loaded successfully")
    except Exception as e:
        raise ValueError(f"Could not load font '{font_path}': {e}")

    if verbose:
        print("Getting font metrics...")
    try:
        head_table = font['head']
        units_per_em = head_table.unitsPerEm
        if verbose:
            print(f"Font units per em: {units_per_em}")
        scale = font_size / units_per_em
        if verbose:
            print(f"Scale factor: {scale}")
    except Exception as e:
        raise RuntimeError(f"Could not get font metrics: {e}")

    return font, scale


def _setup_dxf_document(verbose=False):
    if verbose:
        print("Creating DXF document...")
    try:
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        doc.layers.add("TEXT_OUTLINES", color=1)  # Red color
        if verbose:
            print("DXF document created successfully")
    except Exception as e:
        raise RuntimeError(f"Could not create DXF document: {e}")
    return doc, msp


def _get_font_tables(font, verbose=False):
    if verbose:
        print("Getting glyph information...")
    try:
        glyph_set = font.getGlyphSet()
        if verbose:
            print("Glyph set obtained successfully")
    except Exception as e:
        raise RuntimeError(f"Could not get glyph set: {e}")

    try:
        cmap_subtable = None
        # Prioritize Windows Unicode BMP (3,1) or Symbol (3,0) tables
        for table in font.get('cmap').tables:
            if table.platformID == 3 and table.platEncID in [1, 0]:
                cmap_subtable = table
                if verbose:
                    print(f"Found a suitable cmap table: PlatformID={table.platformID}, PlatEncID={table.platEncID}")
                break
        
        cmap = None
        if cmap_subtable:
            cmap = cmap_subtable.cmap
        else:
            # If no suitable Windows cmap, try to get the best available
            if verbose:
                print("No specific Windows cmap found, trying getBestCmap()")
            cmap = font.getBestCmap()

        if not cmap:
            # If still no cmap, maybe there is another one we can use
            if font.get('cmap').tables:
                if verbose:
                    print("getBestCmap() failed, trying the first available cmap table.")
                cmap = font.get('cmap').tables[0].cmap

        if not cmap:
            raise RuntimeError("No character map found in font")

        if verbose:
            print(f"Character map obtained, {len(cmap)} characters available")
    except Exception as e:
        raise RuntimeError(f"Could not get character map: {e}")

    return glyph_set, cmap


def _get_char_advance(font, glyph_name, glyph_set, scale, spacing, font_size, char, verbose=False):
    x_advance = None
    # Method 1: Try hmtx table
    try:
        if 'hmtx' in font:
            hmtx_table = font['hmtx']
            if verbose:
                print(f"  hmtx table metrics keys: {list(hmtx_table.metrics.keys())[:10]}") # Print first 10 keys
            if glyph_name in hmtx_table.metrics:
                advance_width, _ = hmtx_table.metrics[glyph_name]
                x_advance = (advance_width * scale) * spacing
                if verbose:
                    print(f"  Advanced by {x_advance:.2f} (from hmtx table, width={advance_width})")
    except Exception as e:
        if verbose:
            print(f"  Could not get metrics from hmtx table: {e}")

    # Method 2: Try glyph set width
    if x_advance is None:
        try:
            glyph_obj = glyph_set[glyph_name]
            if hasattr(glyph_obj, 'width') and glyph_obj.width is not None:
                advance_width = glyph_obj.width
                x_advance = (advance_width * scale) * spacing
                if verbose:
                    print(f"  Advanced by {x_advance:.2f} (from glyph set, width={advance_width})")
        except Exception as e:
            if verbose:
                print(f"  Could not get width from glyph set: {e}")

    # Method 3: Fallback to a default advance if other methods fail
    if x_advance is None:
        # Default to a generic advance based on font_size if no other metric is available
        x_advance = (font_size * 0.5) * spacing  # A more generic fallback
        if verbose:
            print(f"  Advanced by {x_advance:.2f} (generic fallback)")

    return x_advance


def _get_kerning_adjustment(font, left_glyph, right_glyph, scale, verbose=False):
    """Get kerning adjustment for a pair of glyphs from the GPOS table."""
    if 'GPOS' not in font:
        return 0

    gpos_table = font['GPOS'].table
    if not hasattr(gpos_table, 'LookupList') or not gpos_table.LookupList:
        return 0

    for lookup in gpos_table.LookupList.Lookup:
        if not lookup or not hasattr(lookup, 'SubTable'):
            continue
        # LookupType 2 is for Pair Adjustment
        if lookup.LookupType == 2:
            for subtable in lookup.SubTable:
                if not subtable or not hasattr(subtable, 'Format'):
                    continue
                # Format 1 is for simple pair kerning
                if subtable.Format == 1:
                    if not hasattr(subtable, 'Coverage') or not hasattr(subtable.Coverage, 'glyphs'):
                        continue
                    
                    try:
                        idx = subtable.Coverage.glyphs.index(left_glyph)
                    except ValueError:
                        continue  # left_glyph not in coverage for this subtable

                    if not hasattr(subtable, 'PairSet') or idx >= len(subtable.PairSet):
                        continue

                    pair_set = subtable.PairSet[idx]
                    if not pair_set or not hasattr(pair_set, 'PairValueRecord'):
                        continue

                    for record in pair_set.PairValueRecord:
                        if record.SecondGlyph == right_glyph:
                            if hasattr(record, 'Value1') and hasattr(record.Value1, 'XAdvance'):
                                kerning_value = record.Value1.XAdvance
                                if verbose:
                                    print(f"  Kerning adjustment for ({left_glyph}, {right_glyph}): {kerning_value * scale:.2f}")
                                return kerning_value * scale
                # Format 2 is for class-based kerning
                elif subtable.Format == 2:
                    # This is more complex and less common for specific pairs.
                    # We will ignore it for now to keep the change simple.
                    pass
    return 0


def add_oval(msp, min_x, min_y, max_x, max_y, offset, verbose=False):
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    x_radius = (width / 2) + offset
    y_radius = (height / 2) + offset

    if verbose:
        print(f"  add_oval: min_x={min_x:.2f}, min_y={min_y:.2f}, max_x={max_x:.2f}, max_y={max_y:.2f}, offset={offset:.2f}")
        print(f"  add_oval: center=({center_x:.2f}, {center_y:.2f}), x_radius={x_radius:.2f}, y_radius={y_radius:.2f}")

    center = (center_x, center_y)

    if x_radius <= 0 or y_radius <= 0:
        if verbose:
            print("  add_oval: Skipping oval creation due to non-positive radius.")
        return

    if x_radius >= y_radius:
        major_axis = (x_radius, 0)
        ratio = y_radius / x_radius
    else:
        major_axis = (0, y_radius)
        ratio = x_radius / y_radius
    
    ellipse_entity = msp.add_ellipse(center, major_axis, ratio)
    ellipse_entity.dxf.layer = "SURROUND"
    if verbose:
        print(f"  add_oval: Added ELLIPSE entity to msp. Layer: {ellipse_entity.dxf.layer}")

def add_double_oval(msp, min_x, min_y, max_x, max_y, offset, verbose=False):
    if verbose:
        print(f"  add_double_oval: Creating outer oval with offset={offset:.2f}")
    # Outer oval
    add_oval(msp, min_x, min_y, max_x, max_y, offset, verbose)
    if verbose:
        print(f"  add_double_oval: Creating inner oval with offset={-offset:.2f}")
    # Inner oval
    add_oval(msp, min_x, min_y, max_x, max_y, -offset, verbose)

def generate_surrounding_shape(msp, min_x, min_y, max_x, max_y, surround, padding, gap, corner_radius, verbose=False):
    """
    Generate paths for the surrounding shape.
    """
    if surround == 'none':
        return

    # Apply padding
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    if verbose:
        print(f"Generating surrounding shape: {surround}")
        print(f"Padding: {padding}, Gap: {gap}, Corner Radius: {corner_radius}")

    def create_rounded_rect(msp, min_x, min_y, max_x, max_y, radius):
        """Create a single rounded rectangle path and add it to msp."""
        if radius == 0:
            # Simple rectangle
            points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
            msp.add_lwpolyline(points, close=True).dxf.layer = "SURROUND"
            return

        # Ensure radius is not larger than half the shortest side
        radius = min(radius, (max_x - min_x) / 2, (max_y - min_y) / 2)
        if radius < 0: radius = 0 # Ensure non-negative radius

        path = []
        
        # Start at the bottom-left straight line segment start
        path.append((min_x + radius, min_y))

        # Bottom line segment
        path.append((max_x - radius, min_y))

        # Bottom-right arc
        path.extend(arc_points((max_x - radius, min_y + radius), radius, 270, 360))

        # Right line segment
        path.append((max_x, min_y + radius))
        path.append((max_x, max_y - radius))

        # Top-right arc
        path.extend(arc_points((max_x - radius, max_y - radius), radius, 0, 90))

        # Top line segment
        path.append((max_x - radius, max_y))
        path.append((min_x + radius, max_y))

        # Top-left arc
        path.extend(arc_points((min_x + radius, max_y - radius), radius, 90, 180))

        # Left line segment
        path.append((min_x, max_y - radius))
        path.append((min_x, min_y + radius))

        # Bottom-left arc
        path.extend(arc_points((min_x + radius, min_y + radius), radius, 180, 270))

        # Close the path by appending the first point
        path.append(path[0])
        msp.add_lwpolyline(path, close=True).dxf.layer = "SURROUND"

    def arc_points(center, radius, start_angle, end_angle, segments=16, reverse=False):
        """Generate points for an arc."""
        points = []
        if reverse:
            start_angle, end_angle = end_angle, start_angle
        
        for i in range(segments + 1):
            angle = math.radians(start_angle + (end_angle - start_angle) * i / segments)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points

    if surround == 'rectangle':
        create_rounded_rect(msp, min_x, min_y, max_x, max_y, corner_radius)
    elif surround == 'double_rectangle':
        create_rounded_rect(msp, min_x, min_y, max_x, max_y, corner_radius)
        # Second rectangle with gap
        min_x_inner = min_x + gap
        min_y_inner = min_y + gap
        max_x_inner = max_x - gap
        max_y_inner = max_y - gap
        
        # Adjust corner radius for inner rectangle
        inner_radius = max(0, corner_radius - gap)
        
        create_rounded_rect(msp, min_x_inner, min_y_inner, max_x_inner, max_y_inner, inner_radius)
    elif surround == 'oval':
        add_oval(msp, min_x, min_y, max_x, max_y, 0, verbose) # Offset 0 for single oval
    elif surround == 'double_oval':
        add_double_oval(msp, min_x, min_y, max_x, max_y, gap, verbose)

    return


def _calculate_line_width(font, line_text, glyph_set, cmap, scale, spacing, font_size, kerning, is_symbol_font, verbose=False):
    """
    Calculates the total width of a single line of text, considering font metrics, spacing, and kerning.
    """
    current_x = 0
    previous_glyph_name = None

    space_advance = 0
    try:
        space_glyph_name = cmap.get(ord(' '))
        if space_glyph_name and 'hmtx' in font:
            space_advance, _ = font['hmtx'][space_glyph_name]
            space_advance *= scale
        else:
            space_advance = font_size * 0.4
    except Exception:
        space_advance = font_size * 0.4

    for char in line_text:
        if char == ' ':
            current_x += space_advance
            previous_glyph_name = None
            continue

        char_code = ord(char)
        if is_symbol_font:
            if char_code <= 255:
                char_code += 0xF000

        if char_code not in cmap:
            previous_glyph_name = None
            continue

        glyph_name = cmap[char_code]

        if kerning and previous_glyph_name:
            adjustment = _get_kerning_adjustment(font, previous_glyph_name, glyph_name, scale, verbose)
            current_x += adjustment

        try:
            glyph = glyph_set[glyph_name]
        except KeyError:
            previous_glyph_name = None
            continue

        x_advance = _get_char_advance(font, glyph_name, glyph_set, scale, spacing, font_size, char, verbose)
        current_x += x_advance
        previous_glyph_name = glyph_name
    return current_x


def text_to_dxf(font_path, lines_of_text, output_path, font_size=20, spacing=1.0, curve_quality=0.5, verbose=False, kerning=True, surround='none', padding=5.0, gap=3.0, corner_radius=0.0, font_index=0, line_spacing=1.5):
    """
    Convert text to DXF outlines using the specified font.
    """
    font, scale = _load_font_and_get_scale(font_path, font_size, font_index, verbose)
    doc, msp = _setup_dxf_document(verbose)
    glyph_set, cmap = _get_font_tables(font, verbose)

    is_symbol_font = False
    if cmap and cmap.keys():
        min_code = min(cmap.keys())
        if 0xF000 <= min_code <= 0xF8FF: # PUA range
            is_symbol_font = True
            if verbose:
                print("Detected a Symbol font based on character codes in the PUA range.")

    space_advance = font_size * 0.4 # Default fallback
    try:
        # Get advance width for space character from hmtx table
        space_glyph_name = cmap.get(ord(' '))
        if space_glyph_name and 'hmtx' in font:
            space_advance, _ = font['hmtx'][space_glyph_name]
            space_advance *= scale
    except Exception:
        pass # Keep fallback value

    if verbose:
        print(f"Space advance width: {space_advance}")
    # First Pass: Calculate widths of all lines
    line_widths = []
    for line_text in lines_of_text:
        line_width = _calculate_line_width(font, line_text, glyph_set, cmap, scale, spacing, font_size, kerning, is_symbol_font, verbose)
        line_widths.append(line_width)
    
    max_line_width = max(line_widths) if line_widths else 0

    # Calculate total text height and initial y_offset for vertical centering
    line_height = font_size * line_spacing
    total_text_height = len(lines_of_text) * line_height - (line_height - font_size) # Adjust for last line not having extra spacing
    
    # Center the entire block of text vertically around y=0
    current_y_offset = -total_text_height / 2 + font_size / 2 # Start from the baseline of the top line

    all_paths = []
    surrounding_paths = []
    successful_chars = 0
    
    # Second Pass: Draw each line
    for line_idx, line_text in enumerate(lines_of_text):
        line_width = line_widths[line_idx]
        
        # Calculate x_offset to center the current line
        x_offset = -line_width / 2 # Center each line horizontally
        
        previous_glyph_name = None  # Reset for kerning for each new line

        for char in tqdm(line_text, desc=f"Processing line {line_idx + 1}/{len(lines_of_text)}", leave=False, disable=not verbose):
            if char == ' ':
                x_offset += space_advance
                previous_glyph_name = None
                continue

            char_code = ord(char)
            if is_symbol_font:
                if char_code <= 255:
                    char_code += 0xF000

            if char_code not in cmap:
                if verbose:
                    print(f"  Warning: Character '{char}' (code: {char_code}) not found in font, skipping")
                previous_glyph_name = None
                continue

            glyph_name = cmap[char_code]

            if kerning and previous_glyph_name:
                adjustment = _get_kerning_adjustment(font, previous_glyph_name, glyph_name, scale, verbose)
                x_offset += adjustment

            try:
                glyph = glyph_set[glyph_name]
            except KeyError:
                if verbose:
                    print(f"  Warning: Glyph '{glyph_name}' not found in font, skipping")
                previous_glyph_name = None
                continue

            pen = DXFPen(msp, x_offset, current_y_offset, scale, curve_quality)
            try:
                glyph.draw(pen)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not draw glyph '{glyph_name}': {e}")
                continue

            pen.endPath()
            pen.draw_to_dxf()
            all_paths.extend(pen.paths)
            successful_chars += 1

            x_advance = _get_char_advance(font, glyph_name, glyph_set, scale, spacing, font_size, char, verbose)
            x_offset += x_advance
            
            previous_glyph_name = glyph_name
        
        current_y_offset -= line_height # Move down for the next line

    if verbose:
        print(f"Processed {successful_chars} characters successfully")

    # After generating all text paths, calculate the bounding box
    if all_paths and surround != 'none':
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for path in all_paths:
            for x, y in path:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        if verbose:
            print(f"Text bounding box: ({min_x:.2f}, {min_y:.2f}) to ({max_x:.2f}, {max_y:.2f})")

        # Generate surrounding shape
        generate_surrounding_shape(msp, min_x, min_y, max_x, max_y, surround, padding, gap, corner_radius, verbose)

    try:
        if output_path:
            # Add a new layer for the surrounding shape
            if surround != 'none':
                doc.layers.add("SURROUND", color=2)  # Yellow color
            
            doc.saveas(output_path)
            print(f"DXF file saved successfully: {output_path}")
        if verbose:
            print("Summary:")
            print(f"  Text: '{' '.join(lines_of_text)}'")
            print(f"  Font: {font_path}")
            print(f"  Font size: {font_size}mm")
            print(f"  Character spacing: {spacing}")
            print(f"  Kerning enabled: {kerning}")
            print(f"  Characters processed: {successful_chars}/{len([c for line in lines_of_text for c in line if c != ' '])}")
    except Exception as e:
        raise RuntimeError(f"Could not save DXF file '{output_path}': {e}")
    finally:
        font.close()
        if verbose:
            print("Font closed successfully")
    return all_paths, msp


def preview_paths(text_paths, msp=None, preview=False, preview_file=None, verbose=False):
    """
    Display or save a preview of the paths using matplotlib.
    """
    if not preview and not preview_file:
        return

    plt.figure()
    for path in text_paths:
        if len(path) > 1:
            x, y = zip(*path)
            plt.plot(x, y, color='black', linewidth=1.5, label='Text Outlines')

    if msp:
        surrounding_paths_for_preview = []
        for entity in msp:
            if verbose:
                print(f"  preview_paths: Checking entity type: {entity.dxftype()}, Layer: {entity.dxf.layer}")
            if entity.dxf.layer == "SURROUND":
                if entity.dxftype() == 'LWPOLYLINE':
                    # Use entity.vertices() for LWPOLYLINE, points are (x, y) tuples
                    points_list = [(x, y) for x, y in entity.vertices()]
                    surrounding_paths_for_preview.append(points_list)
                elif entity.dxftype() == 'ELLIPSE':
                    if verbose:
                        print(f"  preview_paths: Found ELLIPSE on layer {entity.dxf.layer}")
                        print(f"  preview_paths:   Center: {entity.dxf.center}, Major Axis: {entity.dxf.major_axis}, Ratio: {entity.dxf.ratio}")
                        print(f"  preview_paths:   Start Param: {entity.dxf.start_param}, End Param: {entity.dxf.end_param}")

                    # Approximate ellipse with a polyline for preview
                    center = entity.dxf.center
                    major_axis_vec = entity.dxf.major_axis
                    ratio = entity.dxf.ratio
                    start_param = entity.dxf.start_param
                    end_param = entity.dxf.end_param

                    major_radius = major_axis_vec.magnitude
                    minor_radius = major_radius * ratio
                    rotation = major_axis_vec.angle # Angle of major axis with x-axis

                    num_segments = 100
                    ellipse_points = []
                    for i in range(num_segments + 1):
                        angle_param = start_param + (end_param - start_param) * i / num_segments
                        
                        # Point on unrotated ellipse
                        x_unrotated = major_radius * math.cos(angle_param)
                        y_unrotated = minor_radius * math.sin(angle_param)
                        
                        # Rotate and translate
                        x = center.x + x_unrotated * math.cos(rotation) - y_unrotated * math.sin(rotation)
                        y = center.y + x_unrotated * math.sin(rotation) + y_unrotated * math.cos(rotation)
                        
                        ellipse_points.append((x, y))
                    surrounding_paths_for_preview.append(ellipse_points)
        
        for path in surrounding_paths_for_preview:
            if len(path) > 1:
                x, y = zip(*path)
                plt.plot(x, y, color='blue', linestyle='--', linewidth=1.0, label='Surrounding Shape')
    
    plt.axis('equal')
    plt.title('text_to_dxf preview')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    if preview_file:
        plt.savefig(preview_file)
        print(f"Preview saved to {preview_file}")

    if preview:
        plt.show()


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert text to DXF font outlines')
    
    parser.add_argument('--list-fonts', action='store_true',
                       help='List all available system fonts and exit')
    parser.add_argument('--list-common-fonts', action='store_true',
                       help='List common Adobe, Ubuntu, and Microsoft fonts and exit')
    parser.add_argument('text', nargs='*', help='Text string(s) to convert. Each argument represents a new line.')
    parser.add_argument('-o', '--output', type=str, default='output.dxf',
                       help='Output DXF file path (default: output.dxf)')
    parser.add_argument('--line-spacing', type=float, default=1.5,
                       help='Multiplier for vertical spacing between lines (default: 1.5)')
    parser.add_argument('--font', type=str, default='Arial',
                       help='Font name to use (default: Arial) or path to font file')
    parser.add_argument('--font-index', type=int, default=0,
                       help='For .ttc font collections, specify the font index to use (default: 0)')
    parser.add_argument('--size', type=float, default=20, 
                       help='Font size in mm (default: 20)')
    parser.add_argument('--spacing', type=float, default=1.0,
                       help='Character spacing multiplier (default: 1.0)')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='high',
                       help='Curve quality: low, medium, or high (default: high)')
    parser.add_argument('--kerning', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable/disable font kerning (default: --kerning)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose debug output')
    parser.add_argument('--preview', action='store_true',
                          help='Preview the generated paths using matplotlib')
    parser.add_argument('--preview-file', type=str,
                          help='Save the matplotlib preview to a file')

    # Arguments for surrounding shape
    parser.add_argument('--surround', type=str, choices=['none', 'rectangle', 'double_rectangle', 'oval', 'double_oval'], default='none',
                       help='Surround the text with a shape (default: none)')
    parser.add_argument('--padding', type=float, default=5.0,
                       help='Padding between text and surrounding shape in mm (default: 5.0)')
    parser.add_argument('--gap', type=float, default=3.0,
                       help='Gap between double rectangles in mm (default: 3.0)')
    parser.add_argument('--corner-radius', type=float, default=0.0,
                       help='Corner radius for surrounding rectangles in mm (default: 0.0)')
    
    args = parser.parse_args()
    
    # Handle font listing
    if args.list_fonts:
        list_fonts()
        return
    
    if args.list_common_fonts:
        list_common_fonts(args)
        return
    
    # Validate required arguments for text conversion
    if not args.text and not args.list_fonts and not args.list_common_fonts:
        parser.error('text argument is required (unless using --list-fonts or --list-common-fonts)')
    

    
    # Determine font path
    font_path = None
    
    # Check if the font argument is a file path
    if os.path.isfile(args.font) and args.font.lower().endswith(('.ttf', '.otf', '.ttc')):
        # It's a font file path
        font_path = args.font
        if args.verbose:
            print(f"Using font file: {font_path}")
    else:
        # Treat it as a font name and search system fonts
        font_path = find_font_by_name(args.font)
        if not font_path:
            print(f"Could not find font '{args.font}'. Use --list-fonts to see available fonts.")
            sys.exit(1)
        if args.verbose:
            print(f"Using system font: {args.font}")
            print(f"Font path: {font_path}")
    
    quality_map = {
        'low': 0.1,
        'medium': 0.5,
        'high': 1.0
    }
    curve_quality = quality_map[args.quality]
    
    try:
        print(font_path, args.text, args.output, args.size, args.spacing, curve_quality, args.verbose, args.kerning, args.surround, args.padding, args.gap, args.corner_radius, args.font_index, args.line_spacing)
        all_paths, msp = text_to_dxf(font_path, args.text, args.output, args.size, args.spacing, curve_quality, args.verbose, args.kerning, args.surround, args.padding, args.gap, args.corner_radius, args.font_index, args.line_spacing)
        if args.preview or args.preview_file:
            preview_paths(all_paths, msp, args.preview, args.preview_file, args.verbose)
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        sys.exit(1)


all_paths, msp = text_to_dxf('arial.ttf', ["CNC ANUS"], 'output.dxf', 20, 1.0, 1.0, False, True, 'none', 5.0, 3.0, 0.0, 0, 1.5)



# if __name__ == "__main__":
#     # Example usage if run directly
#     if len(sys.argv) == 1:
#         print("Text to DXF Font Outline Generator")
#         print("\nExample usage:")
#         print("  # List all available fonts")
#         print("  python text_to_dxf.py --list-fonts")
#         print("\n  # Basic usage (uses Arial by default)")
#         print("  python text_to_dxf.py \"Hello World\" -o output.dxf")
#         print("\n  # Multi-line text")
#         print("  python text_to_dxf.py \"Hello\" \"World\" -o multi_line.dxf --line-spacing 1.2")
#         print("\n  # Disable kerning")
#         print("  python text_to_dxf.py \"Hello World\" -o output.dxf --no-kerning")
#         print("\n  # Use different system fonts")
#         print("  python text_to_dxf.py \"Hello World\" -o output.dxf --font \"Times New Roman\"")
#         print("  python text_to_dxf.py \"Hello World\" -o output.dxf --font \"Helvetica\" --size 15")
#         print("\n  # Use a specific font file")
#         print("  python text_to_dxf.py \"Hello World\" -o output.dxf --font arial.ttf")
#         print("  python text_to_dxf.py \"Hello World\" -o output.dxf --font LoveDays.ttf --size 25 --verbose")
#         print("\n  # Preview the output without saving a DXF file")
#         print("  python text_to_dxf.py \"Hello World\" --preview")
#         print("\n  # Save the preview to a file")
#         print("  python text_to_dxf.py \"Hello World\" --preview-file preview.png")
#         print("\nRun with --help for more options")
#     else:
#         main()
from pygerber.examples import ExamplesEnum, load_example
from pygerber.gerberx3.api.v2 import GerberFile
from pygerber.gerberx3.api.v2 import ColorScheme, GerberFile, PixelFormatEnum
source_code = load_example(ExamplesEnum.UCAMCO_ex_2_Shapes)
GerberFile.from_file('Gerber_TopLayer.GTL').parse().render_svg("output.svg")
# GerberFile.from_file('Gerber_TopLayer.GTL').parse().render_raster(
#     "output.png",
#     dpmm=100,
#     color_scheme=ColorScheme.COPPER_ALPHA,
#     pixel_format=PixelFormatEnum.RGBA,
# )
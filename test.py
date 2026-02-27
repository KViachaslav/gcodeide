from pygerber.examples import ExamplesEnum, load_example
from pygerber.gerberx3.api.v2 import GerberFile

# source_code = load_example(ExamplesEnum.UCAMCO_ex_2_Shapes)
# print(source_code)
GerberFile.from_file("Gerber_TopLayer.GTL").parse().render_svg("output.svg")
# Запуск
# render_gerber_to_image("Gerber_TopLayer.GTL")

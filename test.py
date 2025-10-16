
# Укажите путь к вашему GBR файлу
file_path = 'copper_l1.gbr'
from pygerber.examples import ExamplesEnum, load_example
from pygerber.gerberx3.api.v2 import GerberFile

# source_code = load_example(file_path)
GerberFile.from_file(file_path).parse().render_svg("output.svg")
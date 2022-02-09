import logging
import re
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def fetch_files(path_source, file_type='.ome.tif', recursive=False):
    """
    Collect all ome.tif files in a list.
    :param file_type:
    :param path_source:
    :param recursive:
    :return: A list of Path to ome.tif files
    """
    pattern = f'*{file_type}'

    path_source = Path(path_source)
    if recursive:
        files_generator = path_source.rglob(pattern)
    else:
        files_generator = path_source.glob(pattern)

    return [file for file in files_generator if not file.name.startswith('.')]


@dataclass
class DataSet:
    path: Path

    @property
    def condition(self):
        return self.path / 'condition.toml'

    @property
    def raw(self):
        """Define the path to raw folder."""
        return self.path / 'raw'

    @property
    def projections(self):
        """Define the path to projections folder."""
        return self.path / 'projections'

    @property
    def predictions(self):
        return self.path / 'predictions.csv'

    @property
    def annotations(self):
        return self.path / 'annotation.csv'

    @property
    def outlines(self):
        return self.path / 'outlines'

    def check_description(self):
        if self.condition.exists():
            logger.info('Description exists')
        else:
            logger.info('Description is missing, do you want to create one?')

    def _check_container(self, name: str, file_type: str):
        """
        Check if the folder `name` exists and whether it contains `file_type` files (recursively)
        :param name:
        :param file_type:
        :return: None
        """
        name = self.path / name

        if not name.exists():
            logger.info('%s is missing', name)
            name.mkdir()
            logger.info('%s has been created', name)
            return

        logger.info('%s exists', name)
        files = [f for f in name.iterdir()]

        if len(files) == 0:
            logger.info(
                '%s is empty, make sure to move the %s files there', name,
                file_type)
            return
        else:
            recursive_files = fetch_files(name, recursive=True,
                                          file_type=file_type)
            logger.info('%s contains %s %s files', name, len(recursive_files),
                        file_type)

    def check_raw(self):
        self._check_container('raw', '.ome.tif')

    def check_projections(self):
        self._check_container('projections', '_max.tif')

    def check_predictions(self):
        raise NotImplementedError

    def check_annotations(self):
        raise NotImplementedError

    def check_outline(self):
        self._check_container('outline', '.png')


@dataclass
class Marker:
    """Represents a marker."""
    protein: str = None
    channel: str = None
    position: int = None
    wave_length: int = None
    code: str = None

    @property
    def _code(self):
        if self.code is not None:
            return self.code
        else:
            return f'{self.channel}{self.protein}{self.wave_length}'

    @classmethod
    def from_code(cls, code, pattern=r'([rgbm])([\w\d]+)', position=None):
        if code is None:
            raise ValueError('Provide a code')
        if code == 'DAPI':
            return cls(protein='DNA',
                       channel='b',
                       position=0
                       )
        else:
            remainder, wave_length = code[:-3], code[-3:]
            res = re.match(pattern, remainder)
            if res is None:
                raise ValueError(f'Regex unsuccessful: {res=}')
            channel, protein = res.groups()
            return cls(protein=protein,
                       channel=channel,
                       position=position,
                       wave_length=wave_length)


@dataclass
class PixelSize:
    value: float
    units: str

    def in_cm(self):
        conversion_map = {
            'um': 10e4,
            'μm': 10e4,
            'nm': 10e7,
            }
        return self.value / conversion_map[self.units]


@dataclass
class Condition:
    genotype: str = 'wt'
    treatment: str = None
    replicate: str = 1
    markers: list = ''
    pixel_size: PixelSize = 1


def get_markers(markers, sep='+'):
    """
    Convert a '+'-delimited string into a list and prepend the DAPI
    :param markers:
    :param sep: delimiter character
    :return: List of markers
    """
    markers_list = markers.split(sep)
    if 'DAPI' not in markers_list:
        markers_list.insert(0, 'DAPI')
    return markers_list


def condition_from_filename(file_name, pattern):
    """
    Extract parameters of dataset.
    :param file_name:
    :param pattern: must contain 4 groups, namely: genotype, treatment, markers, replicate
    :return: Condition object
    """

    pat = re.compile(pattern)
    matched = re.match(pat, file_name)
    if matched is not None:
        genotype, treatment, markers, replicate = matched.groups()
    else:
        raise re.error('no matched element')
    markers_list = get_markers(markers)
    return Condition(genotype=genotype,
                     treatment=treatment,
                     markers=markers_list,
                     replicate=replicate,
                     pixel_size=PixelSize(.1025, 'um'))


def extract_filename(file):
    file_name = file.name
    file_name = file_name.removesuffix(''.join(file.suffixes))
    file_name = file_name.replace('', '')
    file_name = re.sub(r'_(Default|MMStack)_\d-Pos', '', file_name)

    return file_name.replace('', '')

import re
from dataclasses import dataclass


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

import os
import sys
from datetime import datetime
from importlib.metadata import version
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


project = 'CenFind'
author = 'Leo Burgy'
copyright = f'{datetime.now().year}, {author}'
version = version("cenfind")
release = version

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['generated/*']

html_theme = "furo"

html_static_path = []
html_logo = '../../figures/logos/cenfind_logo_full_dark.png'
html_favicon = '../../figures/logos/favicon.ico'
html_theme_options = {
    "sidebar_hide_name": True,
}
html_sidebars = {}

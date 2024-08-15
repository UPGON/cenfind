import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


project = 'CenFind'
copyright = '2023, Leo Burgy'
author = 'Leo Burgy'
version = '0.15.0'
release = version

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['generated/*']

html_theme = "sphinx_rtd_theme"

html_static_path = []
html_logo = '../../figures/logos/cenfind_logo_full_dark.png'
html_favicon = '../../figures/logos/favicon.ico'
html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_button": None,
}
html_sidebars = {}

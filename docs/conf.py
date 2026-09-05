# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyppbox.utils.commontools import getVersionString

project = 'pyppbox'
copyright = '2026, UMONS-Numediart, Ratha SIV'
author = 'Ratha SIV'
version = getVersionString()
release = version

show_authors = True

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_search.extension',
    'myst_parser',
]

autodoc_preserve_defaults = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md', '.gitignore']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

pygments_style = 'sphinx'
master_doc = 'index'

man_pages = [
    (master_doc, 'pyppbox', u'pyppbox Documentation', [author], 1)
]

# Install documentation tools with: python -m pip install -r requirements/docs.txt

htmlhelp_basename = 'pyppboxdocs'
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "🐍📦 pyppbox",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rathaumons/pyppbox",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 3,
    "navigation_with_keys": False,
}

# remove some primary sidebar
html_sidebars = {
    'getstarted': [], 
    'pyppbox/standalone': [], 
    'pyppbox/structure': [], 
    'pyppbox/config': [], 
    'pyppbox/utils': [], 
    'releasenotes': []
}

html_static_path = ['_static']
html_show_sphinx = False
# html_show_sourcelink = False

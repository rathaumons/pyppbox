# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from pyppbox.utils.commontools import getVersionString

project = 'pyppbox'
copyright = '2024, UMONS-Numediart, Ratha SIV'
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

source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}

master_doc = 'index'

man_pages = [
    (master_doc, 'pyppbox', u'pyppbox Documentation', [author], 1)
]

# pip install sphinx myst-parser pydata-sphinx-theme readthedocs-sphinx-search

htmlhelp_basename = 'pyppboxdocs'
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_show_sphinx = False
# html_show_sourcelink = False

html_theme_options = {
    "logo": {
        "text": "🐍📦 pyppbox",
    },
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 3,
}


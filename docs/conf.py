#
# Configuration file for the Sphinx documentation builder.

from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = "recreat"
author = "Sebastian Scheuer"
version = "0.2.5"
release = version

now = datetime.now()
today = f"{now.year}-{now.month:02}-{now.day:02} {now.hour:02}H{now.minute:02}"
copyright = f"{now.year}, {author}"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]
pygments_style = "sphinx"
extensions = [
    "sphinx.ext.intersphinx", 
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme"
]
todo_include_todos = False
html_theme = "sphinx_rtd_theme"


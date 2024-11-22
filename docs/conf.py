#
# Configuration file for the Sphinx documentation builder.

from datetime import datetime

project = "recreat documentation"
author = "Sebastian Scheuer"
version = "0.0.8"
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
html_theme = "furo"
pygments_style = "sphinx"
extensions = ["sphinx.ext.intersphinx", "sphinx_tabs.tabs"]
todo_include_todos = False


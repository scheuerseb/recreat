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
extensions = ["sphinx.ext.intersphinx", "recommonmark", "sphinx_tabs.tabs"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "official_sphinx": ("http://www.sphinx-doc.org/", None),
    "https://gdevops.gitlab.io/tuto_python/": None,
    "https://gdevops.gitlab.io/tuto_django/": None,
    "docker": ("https://gdevops.gitlab.io/tuto_docker/", None),
    "https://gdevops.gitlab.io/tuto_cli/": None,
    "https://gdevops.gitlab.io/tuto_build/": None,
    "https://gdevops.gitlab.io/tuto_kubernetes/": None,
    "http://blockdiag.com/en/": None,
}
extensions = extensions + ["sphinx.ext.todo"]
todo_include_todos = True


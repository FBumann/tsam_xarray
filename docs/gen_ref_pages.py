"""Auto-generate API reference pages."""

import mkdocs_gen_files

# tsam_xarray re-exports everything from __init__.py
# Generate a single API page for the package
with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write("# API Reference\n\n::: tsam_xarray\n")

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.write("- [tsam_xarray](index.md)\n")

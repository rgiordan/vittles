# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = "-W"  # This flag turns warnings into errors.
SPHINXBUILD   = sphinx-build
#SPHINXPROJ    = PackagingScientificPython
SOURCEDIR     = source
BUILDDIR      = build
SOURCEFILES  = $(shell find '../vittles/' -name '*py' | xargs)


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

strip_notebooks:
	nbstripout example_notebooks/*ipynb

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# This target doesn't work, and I'm not sure why.
code_only_html: Makefile
	echo $(SOURCEFILES)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SOURCEFILES) $(SPHINXOPTS)

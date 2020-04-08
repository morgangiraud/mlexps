# Default
all: test
.PHONY: all

CURRENT_DIR=$(shell pwd)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif
ifeq (,$(shell which git))
HAS_GIT=False
else
HAS_GIT=True
endif

## Install conda environment
install:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment.yml
	@echo ">>> Conda env created."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif
ifeq (True,$(HAS_GIT))
	@echo ">>> Cloning numpy stubs."
	git clone git@github.com:numpy/numpy-stubs.git git-deps/.
else
	@echo ">>> Please install git first: brew install git"
endif


## Export conda environment
export_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env export -n mlexps | grep -v "^prefix: " > environment.yml
	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

.PHONY: install export_env

typecheck:
	MYPYPATH=$(CURRENT_DIR)/04-exp-dde-elite mypy $(CURRENT_DIR)/04-exp-dde-elite

lint:
	flake8

yapf:
	yapf --style tox.ini -r -i --exclude 'git-deps/**/*.py' .

test:
	pytest 04-exp-dde-elite/

ci: lint typecheck test

.PHONY: typecheck yapf lint test





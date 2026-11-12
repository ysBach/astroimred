DOCS_DIR := docs/quarto
DOCS_PYTHON ?= $(CURDIR)/.venv-docs/bin/python
DOCS_ENV = PYTHONPATH="$(CURDIR)/src" QUARTO_PYTHON="$(abspath $(DOCS_PYTHON))" MPLBACKEND=module://matplotlib_inline.backend_inline

.PHONY: docs-api docs-render docs-build docs-preview

docs-api:
	cd $(DOCS_DIR) && $(DOCS_ENV) "$(abspath $(DOCS_PYTHON))" -m quartodoc build
	"$(abspath $(DOCS_PYTHON))" $(DOCS_DIR)/_scripts/normalize_api.py $(DOCS_DIR)/api

docs-render: docs-api
	$(DOCS_ENV) quarto render $(DOCS_DIR)

docs-build: docs-render

docs-preview: docs-api
	$(DOCS_ENV) quarto preview $(DOCS_DIR) --no-browser

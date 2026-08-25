SHELL := /bin/bash

.DEFAULT_GOAL := paper
.DELETE_ON_ERROR:

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

PAPER_DIR := paper
SITE_DIR := site
TMP_PDF_DIR := tmp/pdfs

PAPER_NAME := sudoku_padic_regression
PAPER_TEX := $(PAPER_DIR)/$(PAPER_NAME).tex
PAPER_PDF := $(PAPER_DIR)/$(PAPER_NAME).pdf
PAPER_ASSETS := \
	$(PAPER_DIR)/loss_curve.pdf \
	$(PAPER_DIR)/figures/padic_logic_sudoku_solution.png
SITE_PDF := $(SITE_DIR)/$(PAPER_NAME).pdf
SITE_SOURCE_MANIFEST := $(SITE_DIR)/$(PAPER_NAME).sources.sha256

DIST_DIR := dist
SUBMISSION_ZIP := $(DIST_DIR)/$(PAPER_NAME)-source.zip
SUBMISSION_FILES := \
	$(PAPER_NAME).tex \
	loss_curve.pdf \
	figures/padic_logic_sudoku_solution.png

KINDLE_NAME := $(PAPER_NAME)_kindle
KINDLE_TEX := $(TMP_PDF_DIR)/$(KINDLE_NAME).tex
KINDLE_PDF := $(TMP_PDF_DIR)/$(KINDLE_NAME).pdf

RESPONSE_DIR := submission/reviewer-feedback/2026-08-02
RESPONSE_NAME := response_to_referees
RESPONSE_TEX := $(RESPONSE_DIR)/$(RESPONSE_NAME).tex
RESPONSE_PDF := $(RESPONSE_DIR)/$(RESPONSE_NAME).pdf

.PHONY: all paper site submission kindle response clean distclean help

all: paper

help:
	@printf '%s\n' \
		'make paper      Build paper/sudoku_padic_regression.pdf' \
		'make site       Copy the latest paper PDF into site/' \
		'make submission Build the journal source ZIP in dist/' \
		'make kindle     Build the small-page Kindle-sized PDF in tmp/pdfs/' \
		'make response   Build the private response-to-referees PDF' \
		'make clean      Remove LaTeX auxiliary files' \
		'make distclean  Remove auxiliary files and latexmk-managed build outputs'

paper: $(PAPER_PDF)

site: $(SITE_PDF) $(SITE_SOURCE_MANIFEST)

submission: $(SUBMISSION_ZIP)

kindle: $(KINDLE_PDF)

response: $(RESPONSE_PDF)

$(PAPER_PDF): $(PAPER_TEX) $(PAPER_ASSETS)
	cd $(PAPER_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) $(PAPER_NAME).tex

$(SITE_PDF): $(PAPER_PDF)
	mkdir -p $(SITE_DIR)
	cp $(PAPER_PDF) $(SITE_PDF)

$(SITE_SOURCE_MANIFEST): $(PAPER_TEX) $(PAPER_ASSETS)
	mkdir -p $(SITE_DIR)
	shasum -a 256 $(PAPER_TEX) $(PAPER_ASSETS) > $(SITE_SOURCE_MANIFEST)

$(DIST_DIR):
	mkdir -p $(DIST_DIR)

$(SUBMISSION_ZIP): $(PAPER_PDF) $(PAPER_TEX) $(PAPER_ASSETS) | $(DIST_DIR)
	cd $(PAPER_DIR) && zip -q -FS -X ../$(SUBMISSION_ZIP) $(SUBMISSION_FILES)
	unzip -tq $(SUBMISSION_ZIP)

$(TMP_PDF_DIR):
	mkdir -p $(TMP_PDF_DIR)

# Keep the paper source canonical and derive the compact Kindle variant on demand.
$(KINDLE_TEX): $(PAPER_TEX) | $(TMP_PDF_DIR)
	perl -0pe 's|\\usepackage\[a4paper,margin=1in\]\{geometry\}|\\usepackage[paperwidth=4.40in,paperheight=5.94in,margin=0.25in]{geometry}|; s|\\usepackage\{graphicx\}|\\usepackage{graphicx}\n\\graphicspath{{../../paper/}}|; s|\\begin\{document\}|\\setlength{\\emergencystretch}{1.5em}\n\\setlength{\\tabcolsep}{4pt}\n\\AtBeginEnvironment{algorithmic}{\\small}\n\\begin{document}|' $(PAPER_TEX) > $(KINDLE_TEX)

$(KINDLE_PDF): $(KINDLE_TEX) $(PAPER_ASSETS)
	cd $(TMP_PDF_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) $(KINDLE_NAME).tex

$(RESPONSE_PDF): $(RESPONSE_TEX)
	cd $(RESPONSE_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) $(RESPONSE_NAME).tex

clean:
	cd $(PAPER_DIR) && $(LATEXMK) -c $(PAPER_NAME).tex
	if [[ -f $(KINDLE_TEX) ]]; then cd $(TMP_PDF_DIR) && $(LATEXMK) -c $(KINDLE_NAME).tex; fi
	if [[ -f $(RESPONSE_TEX) ]]; then cd $(RESPONSE_DIR) && $(LATEXMK) -c $(RESPONSE_NAME).tex; fi

distclean: clean
	cd $(PAPER_DIR) && $(LATEXMK) -C $(PAPER_NAME).tex
	if [[ -f $(KINDLE_TEX) ]]; then cd $(TMP_PDF_DIR) && $(LATEXMK) -C $(KINDLE_NAME).tex; fi
	if [[ -f $(RESPONSE_TEX) ]]; then cd $(RESPONSE_DIR) && $(LATEXMK) -C $(RESPONSE_NAME).tex; fi

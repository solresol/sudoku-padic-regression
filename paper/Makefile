TEX := sudoku_padic_regression.tex
PDF := $(TEX:.tex=.pdf)
LATEXMK := latexmk
LATEXMKFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf clean distclean

all: pdf

pdf: $(PDF)

$(PDF): $(TEX)
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX)

clean:
	$(LATEXMK) -c $(TEX)

distclean: clean
	rm -f $(PDF)

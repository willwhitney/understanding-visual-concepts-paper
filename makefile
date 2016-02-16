all: html pdf

pdf:
	pandoc main.md \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--template pandoc_template.tex \
		-o output/rendered.pdf

html:
	pandoc main.md \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--template pandoc_template.tex \
		--to html5 \
		-o output/rendered.html --mathjax

watch:
	reload -b -s output/rendered.html & make html & fswatch -o *.md */*.md | xargs -n1 make html

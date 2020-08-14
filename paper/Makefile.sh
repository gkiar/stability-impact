TEX="pdflatex"
BIBTEX="bibtex"
FILE="impact-of-instability"

default:
	${TEX} ${FILE} &&\
	${TEX} ${FILE} &&\
	${BIBTEX} ${FILE} &&\
	${TEX} ${FILE} &&\
	${TEX} ${FILE}

clean:
	rm -f ${FILE}{.blg,.bbl,.log,.aux,.out,.fdb_latexmk,.fls,.synctex.gz} &&\
	rm Makefile

cleanall:
	make clean &&\
	rm -f ${FILE}.pdf

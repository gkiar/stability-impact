TEX="pdflatex"
BIBTEX="bibtex"
FILE="impact-of-instability"
FILE_METHODS="impact-of-instability-methods"

default:
	${TEX} ${FILE} &&\
	${TEX} ${FILE} &&\
	${BIBTEX} ${FILE} &&\
	${TEX} ${FILE} &&\
	${TEX} ${FILE}

clean:
	rm -f ${FILE}{.bcf,.blg,.bbl,.log,.aux,.out,.fdb_latexmk,.fls,.run.xml,.synctex.gz} &&\
	rm Makefile

cleanall:
	make clean &&\
	rm -f ${FILE}.pdf

%.pdf: %.tex 1_polar_E.pdf
	pdflatex $<

1_polar_E.pdf: main.py ImAn.py
	python3 main.py > params.txt

clean:
	rm *~ *.pdf *.aux *.log *.txt

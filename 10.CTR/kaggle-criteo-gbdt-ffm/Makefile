all: gbdt ffm-train ffm-predict

gbdt:
	make -C solvers/gbdt
	ln -sf solvers/gbdt/gbdt

ffm-train:
	make -C solvers/libffm-1.13
	ln -sf solvers/libffm-1.13/ffm-train

ffm-predict:
	make -C solvers/libffm-1.13
	ln -sf solvers/libffm-1.13/ffm-predict

clean:
	rm -f gbdt ffm fc.trva.t10.txt submission.csv *.sp* te.csv tr.csv
	make -C solvers/gbdt clean
	make -C solvers/ffm clean

python ./utils/count.py tr.csv > fc.trva.t10.txt
python ./converters/pre-a.py te.csv te.gbdt.dense te.gbdt.sparse
python ./converters/pre-a.py tr.csv tr.gbdt.dense tr.gbdt.sparse
../gbdt/gbdt -t 30 -s 1 te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse te.gbdt.out tr.gbdt.out
python ./converters/pre-b.py tr.csv tr.gbdt.out tr.ffm
python ./converters/pre-b.py te.csv te.gbdt.out te.ffm
../libffm-1.13/ffm-train -k 4 -t 18 -s 1 -p te.ffm tr.ffm model
../libffm-1.13/ffm-predict te.ffm model te.out
python ./utils/calibrate.py te.out te.out.cal
python ./utils/make_submission.py te.out.cal submission.csv

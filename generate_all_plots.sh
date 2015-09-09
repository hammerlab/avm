python generate_roc_plots.py  --obliteration-years 2 --plot-file roc2.png
python generate_roc_plots.py  --obliteration-years 3 --plot-file roc3.png
python generate_roc_plots.py  --obliteration-years 4 --plot-file roc4.png
python generate_roc_plots.py  --obliteration-years 5 --plot-file roc5.png
python generate_roc_plots.py  --obliteration-years 6 --plot-file roc6.png
python generate_roc_plots.py  --obliteration-years 7 --plot-file roc7.png
python generate_roc_plots.py  --obliteration-years 8 --plot-file roc8.png
python generate_models_vs_years_grid.py --max-obliteration-years 8 --random-forest --gradient-boosting --logistic-regression  --extra-trees --svm --output-file years-vs-model-auc.png

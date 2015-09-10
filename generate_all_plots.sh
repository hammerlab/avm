python generate_test_roc_plots.py --obliteration-years 2 --opacity 0.85 --plot-file nyu-roc2.png --coefs-file nyu-coefs2.txt
python generate_test_roc_plots.py --obliteration-years 3 --opacity 0.85 --plot-file nyu-roc3.png --coefs-file nyu-coefs3.txt
python generate_test_roc_plots.py --obliteration-years 4 --opacity 0.85 --plot-file nyu-roc4.png --coefs-file nyu-coefs4.txt
python generate_test_roc_plots.py --obliteration-years 5 --opacity 0.85 --plot-file nyu-roc5.png --coefs-file nyu-coefs5.txt
python generate_test_roc_plots.py --obliteration-years 6 --opacity 0.85 --plot-file nyu-roc6.png --coefs-file nyu-coefs6.txt
python generate_test_roc_plots.py --obliteration-years 7 --opacity 0.85 --plot-file nyu-roc7.png --coefs-file nyu-coefs7.txt
python generate_test_roc_plots.py --obliteration-years 8 --opacity 0.85 --plot-file nyu-roc8.png --coefs-file nyu-coefs8.txt

python generate_cv_roc_plots.py --obliteration-years 2 --plot-file cv-roc2.png
python generate_cv_roc_plots.py --obliteration-years 3 --plot-file cv-roc3.png
python generate_cv_roc_plots.py --obliteration-years 4 --plot-file cv-roc4.png
python generate_cv_roc_plots.py --obliteration-years 5 --plot-file cv-roc5.png
python generate_cv_roc_plots.py --obliteration-years 6 --plot-file cv-roc6.png
python generate_cv_roc_plots.py --obliteration-years 7 --plot-file cv-roc7.png
python generate_cv_roc_plots.py --obliteration-years 8 --plot-file cv-roc8.png

python generate_models_vs_years_grid.py --max-obliteration-years 8 --random-forest --gradient-boosting --logistic-regression  --extra-trees --svm --output-file years-vs-model-auc.png

./generate_test_plot.sh
./genate_cv_plots.sh
python generate_models_vs_years_grid.py --max-obliteration-years 8 --random-forest --gradient-boosting --logistic-regression  --extra-trees --svm --output-file years-vs-model-auc.png

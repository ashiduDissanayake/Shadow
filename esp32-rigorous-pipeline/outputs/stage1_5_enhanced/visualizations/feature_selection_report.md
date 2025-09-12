# ESP32 Shadow: Feature Selection Analysis Report

**Generated**: 2025-09-12 04:04:10

## Summary Statistics

- **Total Folds**: 15
- **Final Feature Count**: 30
- **Mean F1 Score**: 0.8039 ± 0.1766
- **Mean MCC**: 0.7435
- **ROC AUC**: 0.9533
- **Statistical Significance**: p < 0.000

## Feature Distribution by Signal Type

- **BVP**: 4 features (13.3%)
- **ACC**: 15 features (50.0%)
- **EDA**: 6 features (20.0%)
- **TEMP**: 5 features (16.7%)

## Top 10 Most Stable Features

1. **bvp_BVP_perm_entropy**: 15/15 (100.0% stability)
2. **acc_y_perm_entropy**: 15/15 (100.0% stability)
3. **acc_l2_max**: 15/15 (100.0% stability)
4. **acc_y_lineintegral**: 15/15 (100.0% stability)
5. **acc_z_energy**: 15/15 (100.0% stability)
6. **acc_y_peaks**: 15/15 (100.0% stability)
7. **temp_l2_min**: 14/15 (93.3% stability)
8. **acc_z_rms**: 14/15 (93.3% stability)
9. **acc_z_pct_95**: 14/15 (93.3% stability)
10. **eda_l2_iqr_5_95**: 14/15 (93.3% stability)

## Selected Features (All 30)

1. `bvp_BVP_perm_entropy` - 15/15 (100.0%)
2. `acc_y_perm_entropy` - 15/15 (100.0%)
3. `acc_l2_ptp` - 7/15 (46.7%)
4. `acc_l2_max` - 15/15 (100.0%)
5. `acc_z_peaks` - 12/15 (80.0%)
6. `eda_l2_lineintegral` - 12/15 (80.0%)
7. `acc_l2_peaks` - 7/15 (46.7%)
8. `acc_z_perm_entropy` - 13/15 (86.7%)
9. `acc_y_lineintegral` - 15/15 (100.0%)
10. `eda_EDA_lineintegral` - 12/15 (80.0%)
11. `temp_TEMP_min` - 13/15 (86.7%)
12. `temp_l2_min` - 14/15 (93.3%)
13. `acc_z_rms` - 14/15 (93.3%)
14. `acc_z_min` - 10/15 (66.7%)
15. `acc_z_energy` - 15/15 (100.0%)
16. `acc_z_pct_95` - 14/15 (93.3%)
17. `acc_z_mean` - 10/15 (66.7%)
18. `bvp_l2_iqr` - 8/15 (53.3%)
19. `acc_l2_rms` - 6/15 (40.0%)
20. `eda_l2_iqr_5_95` - 14/15 (93.3%)
21. `acc_y_peaks` - 15/15 (100.0%)
22. `bvp_BVP_n_sign_changes` - 14/15 (93.3%)
23. `eda_EDA_iqr_5_95` - 14/15 (93.3%)
24. `temp_TEMP_energy` - 12/15 (80.0%)
25. `temp_l2_energy` - 12/15 (80.0%)
26. `acc_l2_min` - 8/15 (53.3%)
27. `temp_TEMP_sum` - 5/15 (33.3%)
28. `bvp_l2_peaks` - 9/15 (60.0%)
29. `eda_l2_min` - 5/15 (33.3%)
30. `eda_EDA_max` - 9/15 (60.0%)

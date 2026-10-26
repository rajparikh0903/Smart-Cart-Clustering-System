# SmartCart Customer Segmentation

Unsupervised customer segmentation for an e-commerce platform, using PCA, K-Means and Agglomerative clustering.

## Problem
SmartCart applies the same marketing to every customer. The goal is to group customers by purchasing behaviour, engagement and loyalty so marketing and retention can be targeted.

## Data
- 2,240 customers, 22 attributes: demographics, spend per product category, purchase channels, web visits, recency, complaints, campaign response.
- Source: public Kaggle dataset *Customer Personality Analysis* (Dr. Omar Romero-Hernandez, CC0): https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

## Approach
1. **Cleaning:** median-imputed 24 missing `Income` values; removed 4 outliers (`Age` ≥ 90, `Income` ≥ 600,000), leaving 2,236 rows.
2. **Feature engineering:** `Age`, `Customer_Tenure_Date`, total `Spending`, `TotalChild`; grouped `Education` into 3 levels and `Marital_Status` into `Living_With` (Partner / Alone); one-hot encoded both.
3. **Scaling and PCA:** standardised all features, kept the 11 components that retain 90% of the variance (the first 3 explain only 45%, so they are used for plots only).
4. **Choosing K:** compared the elbow (`KneeLocator`), silhouette, Calinski-Harabasz, Davies-Bouldin and seed stability (ARI across random seeds).
5. **Model comparison:** K-Means vs Agglomerative (Ward) on the same metrics.
6. **Profiling:** per-cluster feature means and income-vs-spending plots.

## Results
| Check | Finding |
|---|---|
| Best K | **2**: highest silhouette (0.197) and Calinski-Harabasz (555); seed stability ≈ 1.00 |
| K ≥ 3 | Seed stability drops to 0.49–0.66, so those partitions are not reproducible |
| K-Means vs Agglomerative | K-Means wins at K=2 (silhouette 0.197 vs 0.138; Calinski-Harabasz 555 vs 307) |

| Segment | Size | Profile | Suggested action |
|---|---|---|---|
| Cluster 0 | 1,262 | Income ≈ 37k, spend ≈ 166, more children, 6.5 web visits/month, 10% campaign response | Discounts, bundles, web promotions |
| Cluster 1 | 974 | Income ≈ 71k, spend ≈ 1,176, buys via store, catalog and web, 22% campaign response | Loyalty and premium offers, retention |

## Limitations
- K=2 is the finest split this feature set supports reproducibly; larger K gave unstable clusters.
- Recency is nearly identical across the two clusters (48.7 vs 49.6 days), so churn-prone customers are not identified by this clustering. They would need a separate Recency-based rule.
- Silhouette scores are low overall (about 0.2), meaning the segments are tendencies rather than sharply separated groups.

## Run it
```bash
pip install pandas numpy seaborn matplotlib scikit-learn kneed jupyter
jupyter notebook Smart_Cart_Clustering_System_fixed.ipynb
```
Place `smartcart_customers.csv` in the same folder as the notebook.

## Files
- `Smart_Cart_Clustering_System.ipynb`: full analysis
- `smartcart_customers.csv`: dataset

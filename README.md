# Digital Marketing Conversion Analysis

This project uses Python to analyze a synthetic digital marketing dataset and answer two questions:

1. Can customer conversion be predicted from demographic, campaign, and engagement data?
2. Can customers be grouped into useful segments based on their behavior?

I used logistic regression for conversion prediction and K-means clustering for customer segmentation. Alongside the baseline models, I tested class weighting and a stricter feature set to see how class imbalance and potentially target adjacent variables affected the results.

## Dataset

The dataset is a synthetic digital marketing dataset from Kaggle containing demographic information, campaign data, engagement metrics, purchase history, and a binary conversion outcome.

Because the data is synthetic, the results should be treated as an example of an analytical workflow rather than evidence about real customers.

Features used in the analysis include:

* Age and gender
* Income
* Campaign channel and type
* Ad spend
* Website visits and pages per visit
* Time on site
* Email opens and clicks
* Previous purchases
* Loyalty points

**Target:** Conversion (1 = converted, 0 = did not convert)

## Tools

* Python
* pandas
* matplotlib
* scikit-learn

## Analysis

### Data preparation and exploration

I started by inspecting the dataset for missing values, duplicates, data types, and unusual distributions. Identifier and constant value columns were removed before modeling, and categorical variables were encoded for use with scikit-learn.

One issue became important immediately: approximately **87.6% of customers were classified as converted**. Because of that imbalance, accuracy alone would give an incomplete picture of model performance.

### Conversion modeling

I trained a baseline logistic regression model and evaluated it using accuracy, precision, recall, F1 score, a confusion matrix, and a classification report.

The baseline model produced:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8912 |
| Precision | 0.8946 |
| Recall    | 0.9929 |
| F1 Score  | 0.9412 |

These results were strong overall, but the model performed substantially better on converted customers than on the minority non-converted class.

#### Class weighted model

I tested a class weighted logistic regression model to see whether giving more weight to the minority class would improve detection of non-converted customers.

It did, but at a significant cost to overall accuracy and performance on converted customers. I retained the baseline model because the primary objective was identifying customers likely to convert, while treating the class weighted model as an important comparison rather than automatically assuming that balancing the classes produced a better model.

#### Stricter feature set

I also trained a version without `ConversionRate` and `ClickThroughRate` because both variables are conceptually close to the conversion outcome.

Performance declined only slightly after removing them, suggesting that the model still contained useful predictive signal from the remaining customer and engagement features.

Among the strongest positive coefficients were:

* Time on site
* Email clicks
* Ad spend
* Previous purchases
* Email opens
* Pages per visit
* Loyalty points
* Website visits

Overall, engagement and previous purchasing behavior were the strongest recurring signals associated with conversion.

### Customer segmentation

For the clustering analysis, I standardized the numeric features and compared candidate values of *k* using both the elbow method and silhouette scores.

**K = 2** produced the highest silhouette score and was used for the final K-means model.

The resulting clusters were similar in size, with one showing somewhat stronger engagement and conversion behavior, including higher pages per visit, time on site, email engagement, and average conversion.

The silhouette scores were low overall, however, so I would not interpret these groups as strongly separated customer personas. The clustering results are more useful as an exploratory segmentation than as evidence of two naturally distinct customer populations.

## Key Findings

The analysis produced several practical takeaways:

* Engagement measures consistently contributed useful conversion signal.
* Previous purchases and loyalty behavior were also associated with conversion.
* High overall model performance can obscure poor performance on a minority class.
* Removing potentially target adjacent features had relatively little effect on predictive performance.
* K-means found some differences between customer groups, but not enough separation to justify treating them as strongly defined segments.

## Limitations

The largest limitation is that the dataset is synthetic. Results should therefore be interpreted as demonstrations of modeling and analysis rather than conclusions about real customer behavior.

The conversion target is also heavily imbalanced, and several marketing variables are conceptually close to the target. I addressed those issues through class weighted and reduced feature comparisons rather than relying only on the baseline model.

Finally, the low silhouette scores indicate that the clustering results provide only modest evidence for distinct customer segments.

## Repository Structure

```text
digital-marketing-conversion-analysis/
├── README.md
├── digital_marketing_conversion_analysis.ipynb
├── requirements.txt
└── data/
    └── digital_marketing_campaign_dataset.csv
```

## Running the Analysis

Clone the repository, install the required Python packages, and run the notebook cells in order.

```bash
pip install -r requirements.txt
```

## Author

J.P. Westerlund

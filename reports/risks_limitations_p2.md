# Risks and limitations (Person 2 draft)

Draft for the Submission 2 executive summary. Michael Font (Person 2); Person 1 integrates.
Numbers come from `notebooks/02_cleaning_p2.ipynb` and `notebooks/05_features_p2.ipynb`.

## For the executive summary (~130 words)

Our findings describe associations, not causes: a listing feature linked to higher ratings will not necessarily raise ratings if changed. We used 2023 product metadata from four Amazon categories, with no review text or dates, so results may not hold for other categories or for current listings. Toys and Games makes up 59% of the modeled products and weighs heavily on the results. Ratings come from self-selected reviewers and skew high (mean 4.27 of 5), which narrows the range the model must predict. We excluded products with fewer than 5 ratings (28.5% of listings) because their averages are unreliable, so the model does not cover new or niche products. About half of products list no price; we filled these with the median and flagged them. Store names are not verified brands.

## Supporting detail

| Risk or limitation | Evidence | Effect on results | How we handled it |
|---|---|---|---|
| Associations, not causation | Observational metadata, no experiment | Cannot say that changing a listing changes its rating | Wording in findings and recommendations |
| Limited scope | 4 categories; 2023 snapshot; no review timestamps | May not generalize to other categories or to today | State scope in every finding |
| Category imbalance | Toys_and_Games is 59.4% of modeled products; All_Beauty 6.5% | Overall metrics mostly reflect Toys | Split stratified by category; report results by category where possible |
| Self-selected, positive ratings | Mean 4.27, median 4.4 of 5 among modeled products | Small spread; errors look small in stars but can still matter | Compare every model with a predict-the-mean baseline |
| Minimum 5 ratings | 469,214 of 1,644,621 products removed (28.5%); 1-rating products have a rating std of 1.48 vs 0.69 at 5-9 | Model does not apply to new or low-volume products | Documented threshold; easy to change via `MIN_RATINGS` |
| Missing prices | 576,830 of 1,175,407 modeled products (49.1%) have no price | Price findings rest on about half of products | Median imputation inside the preprocessor plus a `has_price` flag; no imputation outside training folds |
| Popularity feedback | Rating volume is positively associated with rating | Volume may be partly a result of good ratings, not a cause | Treated as an association only |
| Unverified store labels | 159,052 store names; not verified brands; some "Unknown" | Store effects may mix sellers; new stores get only an average estimate | Cross-fitted target encoding fitted on training data only |
| Data leakage risk | Store encoding and imputation use the target or the data | Leakage would make test scores look better than reality | All fitted steps inside the model pipeline; test set used once |

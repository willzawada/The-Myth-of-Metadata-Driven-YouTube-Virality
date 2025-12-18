<img width="791" height="527" alt="image" src="https://github.com/user-attachments/assets/a580cd4a-a9e8-4b0c-94d4-2ef0956cb0af" />

# Predicting YouTube Viewer Metrics with Machine Learning
For my Machine Learning project, I aim to investigate: *To what extent can video category and metadata (duration, bitrate, frame rate) predict the numerical value of views on YouTube videos using regression models measured by RMSE and R²?* I used [this YouTube dataset from Kaggle](https://www.kaggle.com/datasets/cyberevil545/youtube-videos-data-for-ml-and-trend-analysis), cleaned the data, and conducted exploratory analysis to examine the distribution of engagement metrics. Multiple regression approaches were applied, including Linear Regression as a baseline model, then Random Forest and Gradient Boosting. Engagement variables were highly skewed and weakly correlated with the available metadata, demonstrating limited predictive performance. Gradient Boosting performed the best (R² = 0.144, RMSE = 2.358), though it explained only a small portion of engagement variance. Feature-importance analysis and engagement-ratio comparisons showed that no strong or consistent predictors emerged despite a few relatively higher ratios for Gaming. Overall, the results indicate that basic video metadata provides minimal explanatory power, suggesting that most variation in YouTube engagement is driven by factors not captured in this dataset.

**Overview and Motivation:** YouTube is a social media service provided by Google that has public videos made available for entertainment purposes. Every day I stream YouTube videos, ranging from podcasts, music videos, and movie reactions. I have also noticed that occasionally a new video will go viral and have been very curious about what causes this. With YouTube videos, I aim to investigate supervised learning models to predict engagement on YouTube videos based on metadata features such as video category. Although engagement includes views, likes, and comments, the predictive models in this project focus specifically on predicting view counts, while model performance will be evaluated through RMSE and R² metrics.

**Dataset**: This YouTube Kaggle dataset: https://www.kaggle.com/datasets/cyberevil545/youtube-videos-data-for-ml-and-trend-analysis, contains 17,950 YouTube videos randomly sampled through the YouTube Data API. It includes publicly available metadata within a wide variety of categories, with no copyrighted content. The features of the dataset include:
*   **Identifiers and Descriptions:** Video ID, Title, URL
*   **Video Attributes:** Duration (seconds), Resolution (height and width), Bitrate, Frame Rate, Codec Information
*   **Categorical Metadata:** Video Category, Hashtags
*   **Engagement Metrics:** Views, Likes, Comments

These videos span multiple genres including: Music, Entertainment, Gaming, Pets/Animals, and Technology reviews, ranging from small creators with a few thousand views to major viral hits. 17,950 rows of this data are available for data analysis. I used this full dataset initially for EDA, and then a filtered-clean version (around 15,000-17,000) for training regression models.

With this dataset, I intend to predict video popularity and virality based around certain categories to inform daily users, content creators, and social media analysts from Google, its parent company, for example. Implementing regression models can further enhance predictions and potentially inform recommendation systems.

**Data Cleaning:** The initial inspection of this dataset showed that the column names were consistent and labeled, with no nested lists or dictionaries embedded from within the features. Duplicate rows were checked, and none were found. Afterward, the data types were inventoried. Continuous numerical features included: views, likes, comment count, duration (seconds), height, width, bitrate, and frame rate. While categorical and nominal features included: video category, codec, and hashtags with the text fields, video title and URL being retained for future analysis. Timestamps were correctly stored as object strings and can be converted to datetime for future exploration. A missing-value assessment revealed that there were relatively small proportions of null entries across most columns. Then, the numerical columns were filled with their respective means, and categorical columns were filled with their modes. Once these steps were implemented, the dataset contained no remaining “nulls”. No major formatting errors or out-of-range values were found, and all of the variable types have matched their feature descriptions.

**Exploratory Data Analysis (EDA):** After cleaning the YouTube dataset, I performed exploratory statistical analysis to understand the distribution and relationships among the key variables. I chose to include a small subset of this in the report, though my full EDA can be viewed within the Process Notebook.

**Machine Learning Models Used:** After doing some initial EDA, I narrowed down the scope of my project to focus on using Random Forest and Gradient Boosting models to compare the R² and RMSE measurements of video duration, bitrate, and frame rate (independent X variables) against views per video (the dependent variable). The intent of using both of these regression models is to compare their R² and RMSE measures of accuracy to draw conclusions about how metadata influences viewer engagement. 

Then, to contextualize the performance of the Random Forest and Gradient Boosting models, I added a simple Linear Regression baseline model only as a baseline comparison. I also expanded the predictor set to include encoded video categories to align with prior feedback. Because the primary predictive models in this project were tree-based (Random Forest and Gradient Boosting), the categorical metadata (‘video category’) was encoded using scikit-learn’s LabelEncoder. The video duration, bitrate, frame rate, and category encoded features were kept, while all of the other features were dropped.
Tree-based models handle integer-encoded categories without assuming an ordinal relationship, making LabelEncoder appropriate and computationally efficient. However, no additional encoding (e.g., one-hot encoding) was applied to the baseline model. I also would've done upload hour, however, that data was not available from within this dataset. Throughout these models, I applied a log(y+1) transformation to reduce skewness (since the dataset is large and heavily right-skewed) and stabilize variance. The models were then trained after applying Min/Max scaling to the feature set, followed by splitting the X-values into training/testing data into an 80-20 split. 

Additionally, I included an engagement ratio to lean into the additional insights of how the popularity of viewership is impacted by specific video categories. Lastly, I included three model performance visualizations: feature importances, an actual vs. predicted scatter plot, and a residual plot for the Random Forest and Gradient boosting models. These visualizations help to further understand model behavior and diagnose prediction patterns.

**Results**: After implementing the baseline Linear Regression model, it produced an R² value of 0.017 and an RMSE of 2.528. Then, with the Random Forest model, it produced an R² value of 0.050 and an RMSE of 2.485. Lastly, with the Gradient Boosting model, it produced an R² value of 0.144 and an RMSE of 2.358. Additionally, the engagement ratio summary table shows the ratios of likes to views and comments to views:

<img width="492" height="493" alt="image" src="https://github.com/user-attachments/assets/a2082baa-0f0e-4f2c-8b61-d6b6c8a88b1f" />

Then, to ensure a fair comparison between the Random Forest and Gradient Boosting models, I performed hyperparameter tuning via GridSearchCV for each algorithm. The grids were intentionally kept small to prevent overfitting and to highlight whether model performance could be improved beyond the default settings. Each model was tuned using RMSE-based cross-validation, and the best-performing parameter sets are reported below:

<img width="877" height="104" alt="image" src="https://github.com/user-attachments/assets/39d1fc99-2dac-42fb-86df-16e9bc178e2a" />

I then included three model performance visualizations: feature importances, an actual vs. predicted scatter plot, and a residual plot (Figure 8) for the ensemble Random Forest and Gradient Boosting models. Below, these each help to further understand model behavior and diagnose prediction patterns:

<img width="545" height="374" alt="image" src="https://github.com/user-attachments/assets/7d5240dd-6875-414e-a46f-f1d8adb32066" />

<img width="537" height="374" alt="image" src="https://github.com/user-attachments/assets/16157e18-6a9e-456d-b3be-81281cde3c3d" />

<img width="517" height="518" alt="image" src="https://github.com/user-attachments/assets/adf623d8-bc64-471c-bdb5-22a0bb5d301d" />

<img width="514" height="515" alt="image" src="https://github.com/user-attachments/assets/2da37fa7-7740-4e55-9401-47817df357c8" />

<img width="444" height="454" alt="image" src="https://github.com/user-attachments/assets/6241b967-4f6c-465c-bf19-4575062997cf" />

<img width="451" height="451" alt="image" src="https://github.com/user-attachments/assets/adb97bde-aa0e-4cfb-b04a-07c291fe366d" />

Across all models, train and test performance were very similar, indicating mild underfitting rather than overfitting. This further supports the conclusion that improvements are limited by the available metadata rather than model complexity.

**Final Analysis:**

We can see initially that the engagement metrics were heavily skewed and only weakly related to basic video metadata. This aligned with the poor performance of linear regression (R² = 0.017, RMSE = 2.528) and the diagnostic plots, which showed a consistent underestimation of high-engagement videos. Tree-based models captured more nonlinear structure, but the Random Forest provided only a small improvement (R² = 0.050, RMSE = 2.485), and even Gradient Boosting—the best model, explained just 14% of engagement variance (R² = 0.144) while achieving the lowest prediction error (RMSE = 2.358). While Gradient Boosting improves R² from 0.050 to 0.144, the gain is modest, though cross-validation shows the improvement is consistent across folds.

The RMSE values here represent the average prediction error in log-transformed view counts, so lower scores indicate better accuracy. For example, we see that the log-RMSEs of 2.358 (Gradient-Boosting) implies that the predictions are off by about a factor of 10^2.358 ≈ ~228 on the original scale. Although Gradient Boosting reduced RMSE compared to the baseline, the improvement was modest, reinforcing that the metadata features alone have limited predictive power.

The feature-importance plots indicated that bitrate and duration contributed more to predictions than the other metadata features. Also, the engagement-ratio comparisons showed some category-level differences. For example, Gaming had significantly higher ratios (likes-to-views = 2.34, comments-to-views = 0.77). We also see that shows had higher likes-to-views ratios at 1.09, with comedy at 0.93, and people/blogs at 0.92.

Therefore, across both tree-based models, the largest prediction errors occurred on videos with extremely high view counts. The models systematically underestimated these viral videos, indicating that while they captured general engagement patterns, they failed to model virality spikes. This aligns with the residual plots, which show increasing dispersion at higher predicted values. These errors suggest that viral dynamics are driven by factors absent from the metadata, such as channel influence or external social trends.

**Conclusions and Recommendations:**

These findings suggest that Gaming videos tend to drive video engagement more than most other video categories, with likes tending to receive more engagement than comments. While some categories appear more engaging than others, both category and technical metadata offer limited predictive value. Overall, these predictive models should incorporate richer behavioral and semantic features. The current findings indicate that most variation in YouTube engagement is driven by factors not present in this dataset, such as content quality, audience behavior, channel reputation, and recommendation-based algorithms. This is largely due to missing semantic and behavioral metadata such as channel subscriber count, upload timing, title text embeddings, thumbnail features, and NLP-based attributes.

From a non-technical perspective, these results suggest that content creators and analysts may benefit more from understanding engagement relative to view count rather than attempting to predict raw popularity from metadata. Categories like Gaming consistently show higher likes-to-views and comments-to-views ratios, indicating stronger audience interaction even when total views are not the highest. These insights can help creators prioritize content types that naturally encourage video engagement.

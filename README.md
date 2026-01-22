Premier League Match Predictor

Project Overview:

This project builds a baseline machine learning model to predict Premier League match outcomes (Win / Draw / Loss) using recent team form (points in the last 5 games) and rolling performance statistics.
Data Description
Historical Premier League match data is used, containing fixture dates, competing teams, full-time goals scored, and final match results. From this raw data, team-level performance features are created to reflect recent form and momentum, which are commonly used heuristics in football analysis.
Home and away performances are treated separately, as teams typically have a psychological advantage and perform better at home due to familiar conditions, crowd support, and reduced travel fatigue. This allows the model to learn asymmetric team behaviour depending on venue.
Rolling windows of five matches are used to construct features such as: form, goals scored, conceded and wins in the last five matches. These measure short-term attacking and defensive strength while ensuring that only information available prior to kick-off is used for each fixture.

Training Strategy:

A Random Forest Classifier is used as the baseline model. This model was chosen because it: captures non-linear relationships between features, does not require feature scaling, is relatively robust to correlated inputs, provides feature importance estimates for interpretability.
The dataset is split chronologically without shuffling. The model is trained on past matches and evaluated on future matches, avoiding data leakage that would otherwise inflate performance.

Evaluation, Results and Improvements:
Model performance is evaluated using accuracy, achieving approximately 69% accuracy on the test set. While encouraging for a baseline model, this performance is limited by the simplicity of the feature set and the absence of more expressive match-level information.
Feature importance analysis shows that recent form and goal-scoring metrics dominate predictions, indicating that the model primarily learns a proxy for overall team strength rather than deeper match-specific dynamics.
Future improvements include incorporating expected goals (xG), opponent-adjusted metrics, bookmaker priors, season-by-season training, and probabilistic calibration to improve robustness and realism.

How to run the project:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python merge_features.py
python train_model.py

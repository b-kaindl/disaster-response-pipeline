# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Current Status
This project and the model contained therein still bear quite some room for improvement. I plan to explore this through two main ways:

1. Inclusion of `message source` as predictor.
The idea that certain kinds of messages are dispersed via one source rather than another seems plausible enough to give it a try. The main question here should be whether the improvement in performance is worth the inclusion of additional parameters.

2. Trying different learning algorithms
For now, classification is done using Scikit Learn's `RandomForestClassifier` and NLTK's `TF-IDF`. Trying out different algorithms might provide better results, especially as we're dealing with a highly imbalanced dataset (see section below).

### Considerations on Data and Model Performance

The second graph at the bottom of the webpage clearly shows this imbalance and its impact; When looking at the performance for single labels in terms of the F1 score and their support in the data, which was also used as the criterion for grid search, there is a visible discrepancy between the performance for single labels.
Put into simpler terms, the algorithm seems to tend to choose labels it has seen more often over those that are rare. If we tuned the algorithm purely to achieve a high recall, choosing labels with a large support set becomes a strategy to optimize performance. This is why the grid search was done to optimize w.r.t. the F1 score. As part of 2. In the section above, I also plan to explore F-score that are slightly skewed toward precision to see if this can improve model performance.

# Smart Library System

## AI-Enabled Book Recommendation and Reader Analytics Platform

A Python command-line application that combines library workflows with artificial intelligence and machine-learning modules. Readers can search, borrow, return, review and discover books. Librarians can inspect borrow records, reader personas, recommendation performance, overdue-risk accuracy and reviews.

## Project Summary

The project uses a catalogue of twenty books and reproducible synthetic reader histories. It demonstrates keyword search, borrowing eligibility, TF-IDF content recommendation, collaborative recommendation, popularity-based cold-start handling, hybrid ranking, logistic-regression overdue-risk prediction, K-Means reader clustering and Naive-Bayes review sentiment analysis.

All current data is stored in memory. The project is intended for academic demonstration and is not a production library platform.

## Requirements

- Python 3.9 or newer
- NumPy
- scikit-learn

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scikit-learn
```

## Running the Project

Automated demonstration:

```bash
python smart_library_system.py --demo
```

Interactive mode:

```bash
python smart_library_system.py
```

The automated demo requires no keyboard input. It executes search, eligibility rules, hybrid recommendations, cold-start recommendations, sentiment analysis, borrow records, reader personas, Hit Rate@5 evaluation and overdue-risk accuracy reporting.

## Reader Workflow

After selecting **Continue as Reader**, enter a name using alphabets and spaces. The Reader Menu provides:

| Option | Function |
|---|---|
| 1 | Search books by title, author or genre keyword |
| 2 | Borrow a book after eligibility validation |
| 3 | Return a book and record late status |
| 4 | Get personalised recommendations |
| 5 | Write a book review and analyse sentiment |
| 6 | View reading history and persona |
| 7 | Logout |

A reader cannot borrow when they report overdue books, already hold three active books or have inactive membership. The current CLI treats membership as active for the reader workflow.

## Librarian Workflow

Demo credentials:

Username: librarian
Password: library123

| Option | Function |
|---|---|
| 1 | View all borrow records |
| 2 | View all reviews and sentiment results |
| 3 | View reader persona distribution |
| 4 | View recommendation-engine Hit Rate@5 |
| 5 | View overdue-risk model accuracy |
| 6 | Add a book and refresh recommendation models |
| 7 | Logout |

> These credentials are hard-coded for classroom demonstration only. Replace them with secure hashed authentication before deployment.

## AI/ML Components

| Component | Implementation | Purpose |
|---|---|---|
| Search agent | Keyword term-count scoring | Finds books matching title, author, genre or tags. |
| Content recommender | TF-IDF and cosine similarity | Finds books similar to a reader’s borrowing profile. |
| Collaborative recommender | Reader-book matrix and cosine similarity | Uses similar-reader activity to suggest books. |
| Popularity recommender | History-frequency counts | Provides recommendations for cold-start readers. |
| Hybrid recommender | Normalised weighted combination | Combines content and collaborative signals. |
| Overdue-risk model | Logistic Regression | Predicts a late-return probability. |
| Reader personas | K-Means clustering | Groups readers by borrowing and review features. |
| Sentiment analyser | CountVectorizer and MultinomialNB | Classifies reviews as positive or negative. |

## Recommendation Logic

Readers with a borrowing history receive a hybrid recommendation generated from content and collaborative scores. Readers with no history receive popularity-based recommendations. If similarity signals are insufficient, the system falls back to popularity. The default hybrid weights are equal: 0.5 for content and 0.5 for collaborative recommendations. Previously borrowed books are excluded.

## Model Evaluation

The recommendation engine uses leave-one-out evaluation. For each reader with at least two books, the most recent book is hidden and the engine attempts to recommend it from the remaining history. The result is reported as Hit Rate@5.

The overdue-risk model uses an 80:20 train-test split with a fixed random state. Its test-set accuracy is displayed by the librarian menu and demo mode. These metrics are demonstration results, not production validation.

A verified run produced:

Recommendation method: hybrid (content-based + collaborative)
Recommendation method: popularity (cold-start: no borrowing history yet)
Hit Rate@5: 100.0%  (16 hits out of 16 readers tested)
Overdue Risk Model Accuracy (test set): 68.00%
Demo complete. All AI/ML modules executed successfully for both Reader and Librarian views.

Exact rankings and probabilities can vary with library versions and numerical behaviour, even though random seeds are fixed.

## Project Structure

smart_library_system.py
README.md
Smart_Library_System_Report_Statement.md

The Python source contains the catalogue, in-memory state, CLI menus, recommendation classes, expert-system rules, prediction models, evaluation routines and automated demo.

## Important Classes

- `BookSearchAgent`: keyword-based catalogue search.
- `LibraryExpertSystem`: borrowing eligibility rules.
- `ContentBasedRecommender`: TF-IDF profile similarity.
- `CollaborativeRecommender`: nearest-reader recommendations.
- `PopularityRecommender`: frequency-based fallback.
- `HybridRecommender`: combined ranking model.
- `OverdueRiskModel`: synthetic logistic-regression classifier.
- `ReaderPersonaClustering`: K-Means reader segmentation.
- `SentimentAnalyzer`: review sentiment classifier.
- `SmartLibrarySystem`: interactive workflow and librarian/reader menus.

## Reproducibility

The source fixes `random.seed(42)` and `np.random.seed(42)`. The overdue-risk train-test split and K-Means model also use fixed random states. Bootstrap readers are generated at startup from catalogue genres.

## Data, Privacy and Security

The application does not persist reader histories, borrowing records or reviews. It does not use a database, encryption, secure credential storage, audit logging or role-based session management. Synthetic histories and small built-in review examples are used for training. Do not enter personally identifiable information or use model outputs for high-impact decisions without validation and governance.

## Limitations

The project does not track physical copies, automatic due dates, real fines, reservations, notifications or online catalogue integration. The overdue-risk labels and sentiment corpus are synthetic or very small. Reader and librarian data is lost when the process exits.

## Future Enhancements

- Add a database and repository/service layers.
- Add secure hashed authentication and role-based access control.
- Add physical availability, due dates, fines and reservation queues.
- Train models on anonymised real library data.
- Evaluate recommendations using precision, recall, MAP@K and NDCG.
- Add model calibration, bias checks and monitoring.
- Add web UI, notifications, exports and automated tests.

## Related Report

For the detailed academic report statement, see [`Project_Statement.md`]
(Project_Statement.md).

## License

This project is intended for academic and educational use. Add an institutional or project-specific license before public distribution.




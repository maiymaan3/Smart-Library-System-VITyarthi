# SMART LIBRARY SYSTEM
## Project Statement

### Project Title

**Smart Library System with AI-Based Book Recommendations and Reader Analytics**

### Statement

The Smart Library System is a Python-based command-line application that combines library management functions with artificial intelligence and machine-learning techniques. The system is designed to help readers search for books, borrow and return books, receive personalised recommendations, write reviews and view reading profiles. It also provides librarians with borrowing records, review summaries, reader personas, recommendation evaluation and overdue-risk analytics.

The project uses a catalogue of twenty books containing title, author, genre and descriptive tags. Reader histories are generated reproducibly for demonstration. The recommendation layer includes keyword search, TF-IDF content similarity, collaborative reader similarity, popularity-based cold-start handling and a hybrid recommendation model. The analytics layer includes a logistic-regression overdue-risk model, K-Means reader persona clustering and Multinomial Naive Bayes sentiment classification for reviews.

The application is implemented in Python using NumPy and scikit-learn. It provides both an interactive command-line mode and a fully automated `--demo` mode. The automated demo executes all major AI/ML modules without requiring keyboard input and reports recommendation method, sentiment result, reader persona distribution, Hit Rate@5 and overdue-risk model accuracy.

### Need for the Project

Conventional library software generally focuses on catalogue lookup and loan records. It may not provide personalised discovery or useful behavioural summaries for librarians. This project demonstrates how machine learning can be connected to ordinary library workflows while keeping the implementation understandable for an academic submission.

### Main Objectives

The project aims to implement an intelligent catalogue search, enforce reader borrowing eligibility, generate recommendations for both existing and new readers, predict possible late returns, classify reader behaviour into personas, analyse review sentiment and provide a librarian analytics dashboard.

### Users

The system has two users. A reader searches and borrows books, returns books, requests recommendations, writes reviews and views a reading profile. A librarian views borrow records, reviews, personas and model results and can add a new book to the catalogue.

### Technology Used

| Area | Technology |
|---|---|
| Programming language | Python 3 |
| Numerical computation | NumPy |
| Content features | TF-IDF vectorisation |
| Similarity | Cosine similarity |
| Risk prediction | Logistic Regression |
| Reader clustering | K-Means |
| Sentiment analysis | CountVectorizer and Multinomial Naive Bayes |
| Interface | Command-line menus and argparse |
| Storage | In-memory Python data structures |

### Expected Outcome

The completed system provides an academic demonstration of an AI-enabled library. It produces search results, borrowing decisions, personalised recommendations, sentiment labels, reader personas and measurable model outputs. The project also documents its limitations, including synthetic training data, in-memory storage, fixed demonstration credentials and the absence of production-grade privacy and security controls.

### Execution Command

```bash
python smart_library_system.py --demo
```

### Verified Demonstration

The demo was executed successfully and produced the following results:

```text
Recommendation method: hybrid (content-based + collaborative)
Recommendation method: popularity (cold-start: no borrowing history yet)
Hit Rate@5: 100.0%  (16 hits out of 16 readers tested)
Overdue Risk Model Accuracy (test set): 68.00%
Demo complete. All AI/ML modules executed successfully for both Reader and Librarian views.
```

### Project Boundary

This is an academic prototype. It does not persist personal data, track physical copies, calculate real due dates or provide secure production authentication. Any future deployment should add a database, secure access control, privacy governance, representative training data and proper model validation.

**Source File:** `smart_library_system.py`  
**Prepared By:** "Mohd Aiymaan "  
**Academic Session:** 2026

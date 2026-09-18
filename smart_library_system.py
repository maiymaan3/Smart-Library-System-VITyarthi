

import argparse
import random
import sys

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    print("This project requires scikit-learn. Install with:")
    print("    pip install numpy scikit-learn --break-system-packages")
    sys.exit(1)

random.seed(42)
np.random.seed(42)

LIBRARIAN_USERNAME = "librarian"
LIBRARIAN_PASSWORD = "library123"



BOOKS = [
    {"id": 1, "title": "The Hobbit", "author": "J R R Tolkien", "genre": "Fantasy",
     "tags": "adventure quest dragon magic middle earth journey"},
    {"id": 2, "title": "Harry Potter and the Sorcerers Stone", "author": "J K Rowling", "genre": "Fantasy",
     "tags": "magic school wizard adventure friendship"},
    {"id": 3, "title": "Dune", "author": "Frank Herbert", "genre": "Science Fiction",
     "tags": "desert planet politics prophecy space empire"},
    {"id": 4, "title": "Foundation", "author": "Isaac Asimov", "genre": "Science Fiction",
     "tags": "empire prediction robots future galaxy science"},
    {"id": 5, "title": "The Da Vinci Code", "author": "Dan Brown", "genre": "Mystery",
     "tags": "conspiracy puzzle art history thriller crime"},
    {"id": 6, "title": "Gone Girl", "author": "Gillian Flynn", "genre": "Mystery",
     "tags": "marriage twist crime psychological thriller"},
    {"id": 7, "title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Romance",
     "tags": "classic society marriage england wit love"},
    {"id": 8, "title": "The Notebook", "author": "Nicholas Sparks", "genre": "Romance",
     "tags": "love memory drama emotional relationship"},
    {"id": 9, "title": "Steve Jobs", "author": "Walter Isaacson", "genre": "Biography",
     "tags": "technology innovation apple life story business"},
    {"id": 10, "title": "Educated", "author": "Tara Westover", "genre": "Biography",
     "tags": "memoir family education resilience growth"},
    {"id": 11, "title": "Sapiens", "author": "Yuval Noah Harari", "genre": "History",
     "tags": "humanity evolution civilization society culture"},
    {"id": 12, "title": "Guns Germs and Steel", "author": "Jared Diamond", "genre": "History",
     "tags": "civilization geography anthropology society"},
    {"id": 13, "title": "Atomic Habits", "author": "James Clear", "genre": "Self Help",
     "tags": "habits productivity motivation growth discipline"},
    {"id": 14, "title": "The Power of Habit", "author": "Charles Duhigg", "genre": "Self Help",
     "tags": "habits psychology behavior change routine"},
    {"id": 15, "title": "Clean Code", "author": "Robert Martin", "genre": "Technology",
     "tags": "programming software engineering best practices code"},
    {"id": 16, "title": "Artificial Intelligence A Modern Approach", "author": "Stuart Russell", "genre": "Technology",
     "tags": "ai algorithms search machine learning agents"},
    {"id": 17, "title": "The Alchemist", "author": "Paulo Coelho", "genre": "Fiction",
     "tags": "journey destiny philosophy adventure dream"},
    {"id": 18, "title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction",
     "tags": "justice childhood morality classic society"},
    {"id": 19, "title": "1984", "author": "George Orwell", "genre": "Fiction",
     "tags": "dystopia surveillance government totalitarian control"},
    {"id": 20, "title": "Brave New World", "author": "Aldous Huxley", "genre": "Fiction",
     "tags": "dystopia society technology control future"},
]
BOOK_BY_ID = {b["id"]: b for b in BOOKS}

READER_HISTORY = {}         
ACTIVE_BORROW_COUNT = {}    
BORROW_LOG = []              
REVIEWS = []                 


def bootstrap_synthetic_readers(n_readers=15):
   
    for i in range(1, n_readers + 1):
        name = f"SyntheticReader{i}"
        favorite_genres = random.sample(sorted({b["genre"] for b in BOOKS}), k=2)
        pool = [b["id"] for b in BOOKS if b["genre"] in favorite_genres]
        if len(pool) < 3:
            pool = [b["id"] for b in BOOKS]
        count = random.randint(3, 6)
        READER_HISTORY[name] = random.sample(pool, k=min(count, len(pool)))


bootstrap_synthetic_readers()




class BookSearchAgent:
    def search(self, query, top_n=5):
        terms = [t for t in query.lower().split() if t]
        scored = []
        for b in BOOKS:
            text = f"{b['title']} {b['author']} {b['genre']} {b['tags']}".lower()
            score = sum(text.count(t) for t in terms)
            if score > 0:
                scored.append((b, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]



class LibraryExpertSystem:
    MAX_ACTIVE_BOOKS = 3

    def check_eligibility(self, facts):
        if facts["has_overdue"]:
            return False, "You have overdue books. Please return them before borrowing more."
        if facts["active_borrowed"] >= self.MAX_ACTIVE_BOOKS:
            return False, f"You already hold the maximum of {self.MAX_ACTIVE_BOOKS} books."
        if not facts["membership_active"]:
            return False, "Your membership is not currently active."
        return True, "Eligible to borrow."




class ContentBasedRecommender:
    def __init__(self, books):
        self.books = books
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [f"{b['genre']} {b['genre']} {b['tags']} {b['author']}" for b in books]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.id_to_index = {b["id"]: i for i, b in enumerate(books)}

    def recommend(self, borrowed_ids, top_n=5, exclude=None):
        exclude = set(exclude) if exclude else set(borrowed_ids)
        idxs = [self.id_to_index[bid] for bid in borrowed_ids if bid in self.id_to_index]
        if not idxs:
            return []
        profile_vector = np.asarray(self.tfidf_matrix[idxs].mean(axis=0))
        sims = cosine_similarity(profile_vector, self.tfidf_matrix)[0]
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])
        results = []
        for idx, score in ranked:
            book = self.books[idx]
            if book["id"] in exclude or score <= 0:
                continue
            results.append((book, float(score)))
            if len(results) >= top_n:
                break
        return results




class CollaborativeRecommender:
    def __init__(self, books):
        self.book_ids = [b["id"] for b in books]

    def _build_matrix(self, reader_history):
        readers = list(reader_history.keys())
        matrix = np.zeros((len(readers), len(self.book_ids)))
        for i, r in enumerate(readers):
            for bid in reader_history[r]:
                if bid in self.book_ids:
                    matrix[i, self.book_ids.index(bid)] = 1
        return readers, matrix

    def recommend(self, target_reader, reader_history, top_n=5, k_neighbors=5):
        if target_reader not in reader_history or not reader_history[target_reader]:
            return []
        readers, matrix = self._build_matrix(reader_history)
        target_idx = readers.index(target_reader)
        sims = cosine_similarity(matrix[target_idx:target_idx + 1], matrix)[0]
        sims[target_idx] = -1  # never recommend based on self
        neighbor_idxs = np.argsort(-sims)[:k_neighbors]

        scores = np.zeros(len(self.book_ids))
        for n_idx in neighbor_idxs:
            weight = max(float(sims[n_idx]), 0)
            scores += weight * matrix[n_idx]

        already = set(reader_history[target_reader])
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in ranked:
            bid = self.book_ids[idx]
            if bid in already or score <= 0:
                continue
            results.append((BOOK_BY_ID[bid], float(score)))
            if len(results) >= top_n:
                break
        return results




class PopularityRecommender:
    def recommend(self, reader_history, top_n=5, exclude=None):
        exclude = set(exclude) if exclude else set()
        counts = {}
        for history in reader_history.values():
            for bid in history:
                counts[bid] = counts.get(bid, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        results = []
        for bid, count in ranked:
            if bid in exclude:
                continue
            results.append((BOOK_BY_ID[bid], count))
            if len(results) >= top_n:
                break
        return results




class HybridRecommender:
    def __init__(self, content_model, collab_model, popularity_model,
                 w_content=0.5, w_collab=0.5):
        self.content_model = content_model
        self.collab_model = collab_model
        self.popularity_model = popularity_model
        self.w_content = w_content
        self.w_collab = w_collab

    @staticmethod
    def _normalize(pairs):
        if not pairs:
            return {}
        max_score = max(score for _, score in pairs) or 1
        return {book["id"]: score / max_score for book, score in pairs}

    def recommend(self, reader, reader_history, top_n=5):
        history = reader_history.get(reader, [])
        if not history:
            results = self.popularity_model.recommend(reader_history, top_n=top_n)
            return results, "popularity (cold-start: no borrowing history yet)"

        content_pairs = self.content_model.recommend(history, top_n=20, exclude=set(history))
        collab_pairs = self.collab_model.recommend(reader, reader_history, top_n=20)
        content_scores = self._normalize(content_pairs)
        collab_scores = self._normalize(collab_pairs)

        all_ids = set(content_scores) | set(collab_scores)
        combined = []
        for bid in all_ids:
            score = (self.w_content * content_scores.get(bid, 0)
                     + self.w_collab * collab_scores.get(bid, 0))
            combined.append((bid, score))
        combined.sort(key=lambda x: -x[1])

        if not combined:
            results = self.popularity_model.recommend(reader_history, top_n=top_n, exclude=set(history))
            return results, "popularity (not enough similarity signal yet)"

        results = [(BOOK_BY_ID[bid], score) for bid, score in combined[:top_n]]
        return results, "hybrid (content-based + collaborative)"


def evaluate_recommender_hit_rate(hybrid, reader_history, top_n=5):

    hits, total = 0, 0
    for reader, history in reader_history.items():
        if len(history) < 2:
            continue
        held_out_book = history[-1]
        trial_history = dict(reader_history)
        trial_history[reader] = history[:-1]
        recs, _ = hybrid.recommend(reader, trial_history, top_n=top_n)
        rec_ids = {book["id"] for book, _ in recs}
        total += 1
        if held_out_book in rec_ids:
            hits += 1
    return (hits / total) if total else 0.0, hits, total




class OverdueRiskModel:
    def __init__(self):
        self.model = LogisticRegression()
        self._train()

    def _generate_synthetic_data(self, n=500):
        X, y = [], []
        for _ in range(n):
            books_borrowed_lifetime = np.random.randint(0, 30)
            prior_late_returns = np.random.randint(0, 6)
            genre_diversity = np.random.uniform(0, 1)
            active_books_now = np.random.randint(0, 3)
            score = (0.25 * prior_late_returns - 0.03 * books_borrowed_lifetime
                     - 0.8 * genre_diversity + 0.3 * active_books_now
                     + np.random.normal(0, 1))
            label = 1 if score > 0.5 else 0
            X.append([books_borrowed_lifetime, prior_late_returns, genre_diversity, active_books_now])
            y.append(label)
        return np.array(X), np.array(y)

    def _train(self):
        X, y = self._generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.model.fit(X_train, y_train)
        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))

    def predict_risk(self, books_borrowed_lifetime, prior_late_returns, genre_diversity, active_books_now):
        x = np.array([[books_borrowed_lifetime, prior_late_returns, genre_diversity, active_books_now]])
        return float(self.model.predict_proba(x)[0][1])




class ReaderPersonaClustering:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, n_init=10, random_state=1)

    def _features_for(self, reader, reader_history):
        history = reader_history.get(reader, [])
        total = len(history)
        genres = {BOOK_BY_ID[bid]["genre"] for bid in history if bid in BOOK_BY_ID}
        diversity = len(genres)
        avg_reviews = sum(1 for r in REVIEWS if r[1] == reader)
        return [total, diversity, avg_reviews]

    def assign_personas(self, reader_history):
        readers = list(reader_history.keys())
        X = np.array([self._features_for(r, reader_history) for r in readers])
        if len(readers) < 3:
            return {r: "Not enough data" for r in readers}
        clusters = self.model.fit_predict(X)
        centers = self.model.cluster_centers_[:, 0]  # rank by total books borrowed
        order = np.argsort(centers)
        rank_to_label = {order[0]: "Casual Reader", order[1]: "Regular Reader", order[-1]: "Avid Reader"}
        return {r: rank_to_label.get(c, "Regular Reader") for r, c in zip(readers, clusters)}




class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self._train()

    def _training_corpus(self):
        positive = [
            "this book was gripping and beautifully written",
            "an amazing story with unforgettable characters",
            "i could not put this book down at all",
            "the plot was brilliant and full of surprises",
            "a wonderful and inspiring read for everyone",
            "the writing style was elegant and captivating",
            "highly recommend this book to everyone",
            "the ending was satisfying and well crafted",
            "such a thoughtful and engaging book",
            "one of the best books i have ever read",
        ]
        negative = [
            "the plot was boring and predictable",
            "i found the characters flat and uninteresting",
            "this book was a complete waste of time",
            "the pacing was slow and dragged on forever",
            "very disappointing ending to the story",
            "the writing felt clumsy and hard to follow",
            "i did not enjoy this book at all",
            "too long and repetitive for my taste",
            "the story made no sense by the end",
            "would not recommend this book to anyone",
        ]
        texts = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)
        return texts, labels

    def _train(self):
        texts, labels = self._training_corpus()
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def analyze(self, text):
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        return sentiment, float(proba[pred])




def ask_int(prompt, valid=None):
    while True:
        try:
            val = int(input(prompt))
            if valid is not None and val not in valid:
                print("Invalid option, try again.")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer.")


def ask_yesno(prompt):
    return ask_int(prompt + " (1=Yes, 0=No): ", valid=[0, 1]) == 1


def ask_alpha(prompt):
    """Accepts only alphabets and spaces (e.g. for reader names)."""
    while True:
        value = input(prompt).strip()
        if value and all(word.isalpha() for word in value.split()):
            return value.title()
        print("Invalid input: please enter alphabets only (no numbers or symbols).")


def authenticate_librarian():
    print("\n--- LIBRARIAN LOGIN ---")
    for attempt in range(3):
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        if username == LIBRARIAN_USERNAME and password == LIBRARIAN_PASSWORD:
            print("Login successful.\n")
            return True
        remaining = 2 - attempt
        if remaining > 0:
            print(f"Incorrect username or password. Attempts remaining: {remaining}")
    print("Too many failed attempts. Returning to main menu.\n")
    return False


def print_book_list(pairs, show_score=True):
    for book, score in pairs:
        line = f"  [{book['id']:>2}] {book['title']} - {book['author']} ({book['genre']})"
        if show_score:
            line += f"   match: {score:.2f}"
        print(line)



class SmartLibrarySystem:
    def __init__(self):
        print("Setting up the system, please wait...")
        self.search_agent = BookSearchAgent()
        self.expert_system = LibraryExpertSystem()
        self.content_model = ContentBasedRecommender(BOOKS)
        self.collab_model = CollaborativeRecommender(BOOKS)
        self.popularity_model = PopularityRecommender()
        self.hybrid_model = HybridRecommender(self.content_model, self.collab_model, self.popularity_model)
        self.risk_model = OverdueRiskModel()
        self.persona_model = ReaderPersonaClustering()
        self.sentiment_model = SentimentAnalyzer()
        print("System ready.\n")

   
    def run(self):
        print("=" * 60)
        print(" WELCOME TO THE SMART LIBRARY SYSTEM")
        print("=" * 60)
        while True:
            print("\n1. Continue as Reader")
            print("2. Continue as Librarian")
            print("3. Exit")
            role = ask_int("Enter choice: ", valid=[1, 2, 3])
            if role == 1:
                self.run_reader_menu()
            elif role == 2:
                if authenticate_librarian():
                    self.run_librarian_menu()
            elif role == 3:
                print("Thank you for using the Smart Library System. Goodbye!")
                break

    def run_reader_menu(self):
        name = ask_alpha("Enter your name: ")
        READER_HISTORY.setdefault(name, [])
        while True:
            print(f"\n======= READER MENU ({name}) ========")
            print("1. Search Books")
            print("2. Borrow a Book")
            print("3. Return a Book")
            print("4. Get Book Recommendations")
            print("5. Write a Book Review")
            print("6. View My Reading Profile")
            print("7. Logout")
            choice = ask_int("Enter choice: ", valid=[1, 2, 3, 4, 5, 6, 7])

            if choice == 1:
                self._reader_search()
            elif choice == 2:
                self._reader_borrow(name)
            elif choice == 3:
                self._reader_return(name)
            elif choice == 4:
                self._reader_recommend(name)
            elif choice == 5:
                self._reader_review(name)
            elif choice == 6:
                self._reader_profile(name)
            elif choice == 7:
                print("Logged out.")
                break

    def _reader_search(self):
        query = input("Enter a title, author, or genre keyword: ").strip()
        results = self.search_agent.search(query)
        if not results:
            print("No matching books found.")
            return
        print("\n--- SEARCH RESULTS ---")
        print_book_list(results, show_score=False)

    def _reader_borrow(self, name):
        self._reader_search()
        book_id = ask_int("Enter the ID of the book you want to borrow: ",
                           valid=[b["id"] for b in BOOKS])
        has_overdue = ask_yesno("Do you currently have any overdue books")
        facts = {
            "has_overdue": has_overdue,
            "active_borrowed": ACTIVE_BORROW_COUNT.get(name, 0),
            "membership_active": True,
        }
        eligible, reason = self.expert_system.check_eligibility(facts)
        print(f"\nEligibility check: {reason}")
        if not eligible:
            return

        history = READER_HISTORY.setdefault(name, [])
        history.append(book_id)
        ACTIVE_BORROW_COUNT[name] = ACTIVE_BORROW_COUNT.get(name, 0) + 1
        genres = {BOOK_BY_ID[b]["genre"] for b in history}
        diversity = len(genres) / max(len(history), 1)
        prior_late = sum(1 for rec in BORROW_LOG if rec["reader"] == name and rec["late"])
        risk = self.risk_model.predict_risk(len(history), prior_late, diversity,
                                             ACTIVE_BORROW_COUNT[name])
        BORROW_LOG.append({"reader": name, "book_id": book_id, "returned": False, "late": False})

        print(f"You have borrowed: {BOOK_BY_ID[book_id]['title']}")
        print(f"Predicted late-return risk for this loan: {risk:.1%}")

    def _reader_return(self, name):
        active = [rec for rec in BORROW_LOG if rec["reader"] == name and not rec["returned"]]
        if not active:
            print("You have no active borrowed books.")
            return
        print("\n--- YOUR ACTIVE BORROWS ---")
        for i, rec in enumerate(active, start=1):
            print(f"{i}. {BOOK_BY_ID[rec['book_id']]['title']}")
        idx = ask_int("Select a book number to return: ", valid=list(range(1, len(active) + 1)))
        record = active[idx - 1]
        late = ask_yesno("Was this book returned late")
        record["returned"] = True
        record["late"] = late
        ACTIVE_BORROW_COUNT[name] = max(0, ACTIVE_BORROW_COUNT.get(name, 0) - 1)
        print(f"Returned: {BOOK_BY_ID[record['book_id']]['title']} ({'late' if late else 'on time'})")

    def _reader_recommend(self, name):
        recs, method = self.hybrid_model.recommend(name, READER_HISTORY, top_n=5)
        print(f"\n--- RECOMMENDED FOR YOU (method: {method}) ---")
        if not recs:
            print("Not enough data yet to generate recommendations.")
            return
        print_book_list(recs)

    def _reader_review(self, name):
        book_id = ask_int("Enter the ID of the book you're reviewing: ",
                           valid=[b["id"] for b in BOOKS])
        text = input("Write your review: ").strip()
        if not text:
            print("No review entered.")
            return
        sentiment, confidence = self.sentiment_model.analyze(text)
        REVIEWS.append((BOOK_BY_ID[book_id]["title"], name, text, sentiment, confidence))
        print(f"Review recorded. Detected sentiment: {sentiment.title()} ({confidence:.0%} confidence)")

    def _reader_profile(self, name):
        history = READER_HISTORY.get(name, [])
        print(f"\n--- READING PROFILE: {name} ---")
        if not history:
            print("You haven't borrowed any books yet.")
            return
        print("Books borrowed:")
        for bid in history:
            print(f"  - {BOOK_BY_ID[bid]['title']} ({BOOK_BY_ID[bid]['genre']})")
        personas = self.persona_model.assign_personas(READER_HISTORY)
        print(f"Reader persona: {personas.get(name, 'Not enough data')}")

    def run_librarian_menu(self):
        while True:
            print("\n======= LIBRARIAN MENU ========")
            print("1. View All Borrow Records")
            print("2. View All Reviews")
            print("3. View Reader Persona Overview")
            print("4. View Recommendation Engine Hit-Rate")
            print("5. View Overdue Risk Model Accuracy")
            print("6. Add a New Book to the Catalog")
            print("7. Logout")
            choice = ask_int("Enter choice: ", valid=[1, 2, 3, 4, 5, 6, 7])

            if choice == 1:
                self._librarian_view_borrows()
            elif choice == 2:
                self._librarian_view_reviews()
            elif choice == 3:
                self._librarian_view_personas()
            elif choice == 4:
                self._librarian_view_hit_rate()
            elif choice == 5:
                print(f"\nOverdue Risk Model Accuracy (test set): {self.risk_model.accuracy:.2%}")
            elif choice == 6:
                self._librarian_add_book()
            elif choice == 7:
                print("Logged out.")
                break

    def _librarian_view_borrows(self):
        real_records = [r for r in BORROW_LOG]
        print("\n--- ALL BORROW RECORDS ---")
        if not real_records:
            print("No borrow records yet.")
            return
        for r in real_records:
            status = "Returned (late)" if r["returned"] and r["late"] else \
                     "Returned (on time)" if r["returned"] else "Active"
            print(f"Reader: {r['reader']:<15} Book: {BOOK_BY_ID[r['book_id']]['title']:<45} Status: {status}")

    def _librarian_view_reviews(self):
        print("\n--- ALL BOOK REVIEWS ---")
        if not REVIEWS:
            print("No reviews yet.")
            return
        for title, reviewer, text, sentiment, confidence in REVIEWS:
            print(f"[{title}] by {reviewer}: \"{text}\" -> {sentiment.title()} ({confidence:.0%})")

    def _librarian_view_personas(self):
        personas = self.persona_model.assign_personas(READER_HISTORY)
        real_readers = {r: p for r, p in personas.items() if not r.startswith("SyntheticReader")}
        print("\n--- READER PERSONA OVERVIEW ---")
        counts = {}
        for p in personas.values():
            counts[p] = counts.get(p, 0) + 1
        print(f"Distribution (includes bootstrap training readers): {counts}")
        if real_readers:
            print("Current session readers:")
            for r, p in real_readers.items():
                print(f"  {r:<15} -> {p}")

    def _librarian_view_hit_rate(self):
        rate, hits, total = evaluate_recommender_hit_rate(self.hybrid_model, READER_HISTORY, top_n=5)
        print("\n--- RECOMMENDATION ENGINE EVALUATION ---")
        print("Method: Leave-one-out testing - each reader's most recent book")
        print("is hidden, then we check if the recommender predicts it back.")
        print(f"Hit Rate@5: {rate:.1%}  ({hits} hits out of {total} readers tested)")

    def _librarian_add_book(self):
        new_id = max(b["id"] for b in BOOKS) + 1
        title = input("Enter book title: ").strip()
        author = ask_alpha("Enter author name: ")
        genre = input("Enter genre: ").strip().title()
        tags = input("Enter a few descriptive tags (space separated): ").strip()
        BOOKS.append({"id": new_id, "title": title, "author": author, "genre": genre, "tags": tags})
        BOOK_BY_ID[new_id] = BOOKS[-1]
        # Recommenders that index the catalog need to be rebuilt after a change
        self.content_model = ContentBasedRecommender(BOOKS)
        self.collab_model = CollaborativeRecommender(BOOKS)
        self.hybrid_model = HybridRecommender(self.content_model, self.collab_model, self.popularity_model)
        print(f"Added '{title}' to the catalog with ID {new_id}. Recommendation models refreshed.")

    def run_demo(self):
        print("=" * 60)
        print(" RUNNING FULLY AUTOMATED DEMO (no keyboard input required)")
        print("=" * 60)

        print("\n>>> Reader 'Meera' (existing interests: Fantasy) <<<")
        READER_HISTORY["Meera"] = [1, 2]  # The Hobbit, Harry Potter
        recs, method = self.hybrid_model.recommend("Meera", READER_HISTORY, top_n=5)
        print(f"Recommendation method: {method}")
        print_book_list(recs)

        print("\n>>> Reader 'Arjun' (new reader, no history - cold start) <<<")
        READER_HISTORY.setdefault("Arjun", [])
        recs, method = self.hybrid_model.recommend("Arjun", READER_HISTORY, top_n=5)
        print(f"Recommendation method: {method}")
        print_book_list(recs)

        print("\n>>> Borrow eligibility check for 'Meera' <<<")
        facts = {"has_overdue": False, "active_borrowed": 1, "membership_active": True}
        eligible, reason = self.expert_system.check_eligibility(facts)
        print(f"Result: {reason}")

        print("\n>>> Sample book search: 'space adventure' <<<")
        print_book_list(self.search_agent.search("space adventure"), show_score=False)

        print("\n>>> Sample book reviews (sentiment analysis) <<<")
        sample_reviews = [
            ("The Hobbit", "Meera", "this book was gripping and beautifully written"),
            ("Dune", "Arjun", "the pacing was slow and dragged on forever"),
        ]
        for title, reviewer, text in sample_reviews:
            sentiment, confidence = self.sentiment_model.analyze(text)
            REVIEWS.append((title, reviewer, text, sentiment, confidence))
            print(f"[{title}] \"{text}\" -> {sentiment.title()} ({confidence:.0%} confidence)")

        BORROW_LOG.append({"reader": "Meera", "book_id": 1, "returned": True, "late": False})
        BORROW_LOG.append({"reader": "Arjun", "book_id": 3, "returned": False, "late": False})

        print("\n" + "=" * 60)
        print(" LIBRARIAN DASHBOARD (auto-demo - login skipped for headless run)")
        print("=" * 60)
        self._librarian_view_borrows()
        self._librarian_view_reviews()
        self._librarian_view_personas()
        self._librarian_view_hit_rate()
        print(f"\nOverdue Risk Model Accuracy (test set): {self.risk_model.accuracy:.2%}")
        print("\nDemo complete. All AI/ML modules executed successfully "
              "for both Reader and Librarian views.")




def main():
    parser = argparse.ArgumentParser(
        description="Smart Library System with AI Book Recommendation (CSA2001 project)")
    parser.add_argument("--demo", action="store_true",
                         help="Run a fully automated, non-interactive demo "
                              "that exercises every AI/ML module.")
    args = parser.parse_args()

    system = SmartLibrarySystem()
    if args.demo:
        system.run_demo()
    else:
        system.run()


if __name__ == "__main__":
    main()

from src.analyzer import IMDBReviewAnalyzer

analyzer = IMDBReviewAnalyzer()
analyzer.train_sentiment()

review = "The movie had amazing acting but the plot was boring."
print(analyzer.analyze(review))

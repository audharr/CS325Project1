import os
import subprocess
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze_sentiment(self, comment):
        """Analyze the sentiment of a comment."""
        pass

class CommentReader(ABC):
    @abstractmethod
    def read_comments(self, input_file):
        """Read comments from the given file."""
        pass


class LocalLLMSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, model_name):
        self.model_name = model_name

    def analyze_sentiment(self, comment):
        query = (
            f'Please respond using only one word: positive, negative, or neutral. '
            f'Is this comment "{comment}" positive, negative, or neutral?'
        )
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=query,
                text=True,
                capture_output=True,
                check=True
            )
            output = result.stdout.strip().lower()
            for sentiment in ["positive", "negative", "neutral"]:
                if sentiment in output:
                    return sentiment
            return "unknown"
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] LLM processing failed for comment: {comment}\n{e}")
            return "unknown"


class FileCommentReader(CommentReader):
    def read_comments(self, input_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                comments = [line.strip() for line in file.readlines() if line.strip()]
            print(f"[DEBUG] Successfully read {len(comments)} comments from {input_file}")
            return comments
        except FileNotFoundError:
            print(f"[ERROR] File not found: {input_file}")
            return []
        except Exception as e:
            print(f"[ERROR] An error occurred while reading the file: {input_file}\n{e}")
            return []


class CommentProcessor:
    def __init__(self, comment_reader: CommentReader, sentiment_analyzer: SentimentAnalyzer):
        self.comment_reader = comment_reader
        self.sentiment_analyzer = sentiment_analyzer

    def process_comments(self, input_file, output_file):
        comments = self.comment_reader.read_comments(input_file)
        if not comments:
            print(f"[DEBUG] No comments found in {input_file}. Skipping...")
            return

        sentiments = []
        print(f"[DEBUG] Processing {len(comments)} comments from {input_file}")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for comment in comments:
                sentiment = self.sentiment_analyzer.analyze_sentiment(comment)
                sentiments.append(sentiment)
                outfile.write(sentiment + '\n')
                print(f"[DEBUG] Extracted sentiment: {sentiment}")
        return sentiments


class BatchCommentProcessor:
    def __init__(self, input_dir, output_dir, comment_processor: CommentProcessor):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.comment_processor = comment_processor
        self.device_sentiments = {}

    def process_all_files(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for file_name in os.listdir(self.input_dir):
            if file_name.endswith(".txt"):  # Only process .txt files
                device_name = file_name.replace("_comments.txt", "")
                input_file = os.path.join(self.input_dir, file_name)
                output_file = os.path.join(
                    self.output_dir, file_name.replace("_comments", "_sentiments")
                )
                print(f"[DEBUG] Processing file: {input_file}")
                sentiments = self.comment_processor.process_comments(input_file, output_file)
                self.device_sentiments[device_name] = sentiments

        return self.device_sentiments

class SentimentPlotter:
    @staticmethod
    def plot_sentiments(device_sentiments, output_path):
        for device, sentiments in device_sentiments.items():
            counts = {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            }
            plt.bar(counts.keys(), counts.values())
            plt.title(f"Sentiment Distribution for {device}")
            plt.xlabel("Sentiments")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_path, f"{device}_sentiment_plot.png"))
            plt.clf()


if __name__ == "__main__":
    input_dir = "./Comments"
    output_dir = "./Processed"
    plots_dir = "./Plots"
    model_name = "phi3"

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Dependency injection
    comment_reader = FileCommentReader()
    sentiment_analyzer = LocalLLMSentimentAnalyzer(model_name=model_name)
    comment_processor = CommentProcessor(comment_reader, sentiment_analyzer)
    batch_processor = BatchCommentProcessor(input_dir, output_dir, comment_processor)

    print("[DEBUG] Starting processing...")
    device_sentiments = batch_processor.process_all_files()

    print("[DEBUG] Generating plots...")
    SentimentPlotter.plot_sentiments(device_sentiments, plots_dir)
    print("[DEBUG] Processing and plotting completed.")

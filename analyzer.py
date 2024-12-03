import os
import subprocess
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging
import numpy as np


# Configure logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Abstract base class for sentiment analysis
class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze_sentiment(self, comment):
        """Analyze the sentiment of a comment."""
        pass


# Abstract base class for reading comments
class CommentReader(ABC):
    @abstractmethod
    def read_comments(self, input_file):
        """Read comments from the given file."""
        pass


# Implementation for analyzing sentiment using a local LLM
class LocalLLMSentimentAnalyzer(SentimentAnalyzer):
    VALID_SENTIMENTS = {"positive", "negative", "neutral"}

    def __init__(self, model_name):
        self.model_name = model_name

    def analyze_sentiment(self, comment):
        """
        Analyze the sentiment of a single comment using an external LLM model.
        Returns "positive", "negative", or "neutral".
        """
        query = f'Please respond using only one word: positive, negative, or neutral. Is this comment "{comment}" positive, negative, or neutral?'

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=query,
                capture_output=True,
                text=True,
            )

            if result.stderr:
                logging.error(f"Model error: {result.stderr.strip()}")
                return "neutral"  # Default to neutral in case of model error

            sentiment = result.stdout.strip().lower()
            if sentiment in self.VALID_SENTIMENTS:
                return sentiment

            logging.warning(f"Unexpected output: {result.stdout.strip()}")
            return "unknown"  # Handle unexpected outputs gracefully

        except subprocess.SubprocessError as e:
            logging.error(f"Error in sentiment analysis subprocess: {e}")
            return "neutral"
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return "neutral"


# Implementation for reading comments from a file
class FileCommentReader(CommentReader):
    def read_comments(self, input_file):
        """
        Reads comments from a text file, ensuring only valid lines are processed.
        Stops reading after 20 comments to improve efficiency.
        """
        comments = []
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= 20:  # Cap to 20 comments (with permission)
                        break
                    line = line.strip()
                    if line:  # Ignore empty lines
                        comments.append(line)
            logging.info(f"Read {len(comments)} comments from {input_file}")
            return comments
        except FileNotFoundError:
            logging.error(f"File not found: {input_file}")
            return []
        except UnicodeDecodeError as e:
            logging.error(f"Unicode error reading {input_file}: {e}")
            return []
        except Exception as e:
            logging.error(f"Error reading {input_file}: {e}")
            return []


# Class for processing comments with a sentiment analyzer
class CommentProcessor:
    def __init__(self, comment_reader: CommentReader, sentiment_analyzer: SentimentAnalyzer):
        self.comment_reader = comment_reader
        self.sentiment_analyzer = sentiment_analyzer

    def process_comments(self, input_file, output_file):
        """
        Processes comments from a file, performs sentiment analysis, and writes results to an output file.
        """
        comments = self.comment_reader.read_comments(input_file)
        if not comments:
            logging.warning(f"No comments to process in {input_file}.")
            return []

        sentiments = []
        try:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for comment in comments:
                    sentiment = self.sentiment_analyzer.analyze_sentiment(comment)
                    sentiments.append(sentiment)
                    outfile.write(sentiment + '\n')
            logging.info(f"Processed {len(comments)} comments from {input_file} to {output_file}")
        except IOError as e:
            logging.error(f"Error writing to {output_file}: {e}")
        return sentiments


# Class for batch processing of comments
class BatchCommentProcessor:
    def __init__(self, input_dir, output_dir, comment_processor: CommentProcessor):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.comment_processor = comment_processor
        self.device_sentiments = {}

    def process_all_files(self):
        """
        Processes all .txt files in the input directory and stores sentiment results.
        """
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

        text_files = [f for f in os.listdir(self.input_dir) if f.endswith(".txt")]
        if not text_files:
            logging.error(f"No .txt files found in {self.input_dir}")
            return {}

        for file_name in text_files:
            device_name = file_name.replace("_comments.txt", "")
            input_file = os.path.join(self.input_dir, file_name)
            output_file = os.path.join(
                self.output_dir, file_name.replace("_comments", "_sentiments")
            )
            logging.info(f"Processing {input_file}...")
            sentiments = self.comment_processor.process_comments(input_file, output_file)
            self.device_sentiments[device_name] = sentiments

        return self.device_sentiments


# Class for plotting sentiment results
class SentimentPlotter:
    @staticmethod
    def plot_sentiments(device_sentiments, output_file):
        """
        Plots sentiment distribution for all devices in a single file and increments y-axis by 5.
        
        Args:
        - device_sentiments: dict where keys are device names and values are lists of sentiments.
        - output_file: str, path to save the combined sentiment plot.
        """
        # Initialize the figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar width and positions
        devices = list(device_sentiments.keys())
        sentiment_types = ["positive", "negative", "neutral"]
        bar_width = 0.2
        x_indices = np.arange(len(devices))

        # Calculate sentiment counts for each device
        sentiment_counts = {sentiment: [] for sentiment in sentiment_types}
        for device in devices:
            sentiments = device_sentiments[device]
            sentiment_counts["positive"].append(sentiments.count("positive"))
            sentiment_counts["negative"].append(sentiments.count("negative"))
            sentiment_counts["neutral"].append(sentiments.count("neutral"))

        # Plot each sentiment type
        for i, sentiment in enumerate(sentiment_types):
            ax.bar(
                x_indices + i * bar_width,
                sentiment_counts[sentiment],
                width=bar_width,
                label=sentiment,
                color=["blue", "red", "yellow"][i],
            )

        # Configure the plot
        ax.set_xticks(x_indices + bar_width)
        ax.set_xticklabels(devices)
        ax.set_ylabel("Count")
        ax.set_xlabel("Devices")
        ax.set_title("Sentiment Distribution Across Devices")
        ax.legend()
        
        # Adjust y-axis ticks to increment by 5
        max_count = max([max(counts) for counts in sentiment_counts.values()])
        ax.set_yticks(range(0, max_count + 6, 5))

        # Save the plot
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        logging.info(f"Saved combined sentiment plot at {output_file}")
        plt.close()

# main execution
if __name__ == "__main__":
    input_dir = "./Comments"
    output_dir = "./Processed"
    plots_dir = "./Plots"
    model_name = "phi3"

    comment_reader = FileCommentReader()
    sentiment_analyzer = LocalLLMSentimentAnalyzer(model_name)
    comment_processor = CommentProcessor(comment_reader, sentiment_analyzer)
    batch_processor = BatchCommentProcessor(input_dir, output_dir, comment_processor)

    logging.info("Starting batch processing...")
    device_sentiments = batch_processor.process_all_files()

    logging.info("Generating plots...")
    SentimentPlotter.plot_sentiments(device_sentiments, plots_dir)

    logging.info("Processing and plotting completed.")
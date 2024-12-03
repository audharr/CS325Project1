import os
import subprocess
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    VALID_SENTIMENTS = ["positive", "negative", "neutral"]

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
                capture_output=True,
                text=True,
            )

            if result.stderr:
                logging.error(f"Model returned an error: {result.stderr.strip()}")
                return "neutral"

            sentiment = result.stdout.strip().lower()
            if sentiment in self.VALID_SENTIMENTS:
                return sentiment

            logging.warning(f"Unexpected output: {result.stdout.strip()}")
            return "unknown"

        except Exception as e:
            logging.error(f"Unexpected error during sentiment analysis: {e}")
            return "neutral"


class FileCommentReader(CommentReader):
    def read_comments(self, input_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                comments = []
                for i, line in enumerate(file):
                    if i >= 40:  # Stops reading after 40 comments
                        break
                    line = line.strip()
                    if line:  # Skip empty lines
                        comments.append(line)
            logging.info(f"Successfully read {len(comments)} comments from {input_file}")
            return comments
        except FileNotFoundError:
            logging.error(f"File not found: {input_file}")
            return []
        except UnicodeDecodeError as e:
            logging.error(f"Unicode error while reading {input_file}: {e}")
            return []
        except Exception as e:
            logging.error(f"An error occurred while reading {input_file}: {e}")
            return []


class CommentProcessor:
    def __init__(self, comment_reader: CommentReader, sentiment_analyzer: SentimentAnalyzer):
        self.comment_reader = comment_reader
        self.sentiment_analyzer = sentiment_analyzer

    def process_comments(self, input_file, output_file):
        comments = self.comment_reader.read_comments(input_file)
        if not comments:
            logging.info(f"No comments found in {input_file}. Skipping...")
            return []

        sentiments = []
        logging.info(f"Processing {len(comments)} comments from {input_file}")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for comment in comments:
                sentiment = self.sentiment_analyzer.analyze_sentiment(comment)
                sentiments.append(sentiment)
                outfile.write(sentiment + '\n')
                logging.debug(f"Extracted sentiment: {sentiment}")
        return sentiments


class BatchCommentProcessor:
    def __init__(self, input_dir, output_dir, comment_processor: CommentProcessor):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.comment_processor = comment_processor
        self.device_sentiments = {}

    def process_all_files(self):
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Get list of all .txt files in the input directory
        text_files = [
            file_name for file_name in os.listdir(self.input_dir)
            if file_name.endswith(".txt")
        ]

        if not text_files:
            logging.error(f"No .txt files found in {self.input_dir}")
            return {}

        # Process each file
        for file_name in text_files:
            device_name = file_name.replace("_comments.txt", "")
            input_file = os.path.join(self.input_dir, file_name)
            output_file = os.path.join(
                self.output_dir, file_name.replace("_comments", "_sentiments")
            )
            logging.info(f"Processing file: {input_file}")
            sentiments = self.comment_processor.process_comments(input_file, output_file)
            self.device_sentiments[device_name] = sentiments

        return self.device_sentiments


class SentimentPlotter:
    @staticmethod
    def plot_sentiments(device_sentiments, output_path):
        os.makedirs(output_path, exist_ok=True)

        # Set up the figure and bar width
        devices = list(device_sentiments.keys())
        x = range(len(devices))  # X-axis positions for devices
        width = 0.2  # Bar width

        # Initialize sentiment counts for plotting
        positive_counts = [device_sentiments[device].count("positive") for device in devices]
        negative_counts = [device_sentiments[device].count("negative") for device in devices]
        neutral_counts = [device_sentiments[device].count("neutral") for device in devices]

        # Create a grouped bar plot
        plt.bar([pos - width for pos in x], positive_counts, width, label='Positive', color='blue')
        plt.bar(x, negative_counts, width, label='Negative', color='red')
        plt.bar([pos + width for pos in x], neutral_counts, width, label='Neutral', color='yellow')

        # Add labels, title, and legend
        plt.xlabel("Devices")
        plt.ylabel("Count")
        plt.title("Sentiment Distribution for All Devices")
        plt.xticks(x, devices, rotation=45, ha='right')  # Label devices on the x-axis
        plt.legend()

        # Customize y-axis ticks to increment by 5
        max_y = max(max(positive_counts), max(negative_counts), max(neutral_counts))
        plt.yticks(range(0, max_y + 6, 5))  # Increment by 5, ensuring room for highest count

        # Save the figure
        plt.tight_layout()
        output_file = os.path.join(output_path, "sentiment_distribution_all_devices.png")
        plt.savefig(output_file)
        plt.clf()


if __name__ == "__main__":
    input_dir = "./Comments"
    output_dir = "./Processed"
    plots_dir = "./Plots"
    model_name = "phi3"

    # Dependency injection
    comment_reader = FileCommentReader()
    sentiment_analyzer = LocalLLMSentimentAnalyzer(model_name=model_name)
    comment_processor = CommentProcessor(comment_reader, sentiment_analyzer)
    batch_processor = BatchCommentProcessor(input_dir, output_dir, comment_processor)

    logging.info("Starting processing...")
    device_sentiments = batch_processor.process_all_files()

    logging.info("Generating a single plot for all devices...")
    SentimentPlotter.plot_sentiments(device_sentiments, plots_dir)
    logging.info("Processing and plotting completed.")
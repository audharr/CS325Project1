import os         # inirialize the figure and axes              # Provides way to interact with os; handles file paths
import subprocess                       # Manage external processes
import matplotlib.pyplot as plt         # Plotting graphs
from abc import ABC, abstractmethod     # Defining abstract bases
import logging                          # Provides flexible framework
import numpy as np                      # Library for numerical computation


# Configure logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentAnalyzer(ABC):                           # Abstract base class for sentiment analysis
    @abstractmethod
    def analyze_sentiment(self, comment):
        """Analyze the sentiment of a comment."""
        pass

class CommentReader(ABC):                               # Abstract base class for reading commnets
    @abstractmethod
    def read_comments(self, input_file):
        """Read comments from the given file."""
        pass

class LocalLLMSentimentAnalyzer(SentimentAnalyzer):                         # Implementation for analyzing sentiment using a LLM
    VALID_SENTIMENTS = {"positive", "negative", "neutral"}                  # Positive, Negative, and Neutral

    def __init__(self, model_name):
        self.model_name = model_name

    def analyze_sentiment(self, comment):
        """
        Analyze the sentiment of a single comment using an external LLM model.
        Returns "positive", "negative", or "neutral".
        """
        query = f'Please respond using only one word: positive, negative, or neutral. Is this comment "{comment}" positive, negative, or neutral?'

        try:
            result = subprocess.run(                    # executes the LLM using subprocess
                ["ollama", "run", self.model_name],     # runs model that is initialized
                input=query,                            # send to model
                capture_output=True,                    # capture models response
                text=True,                              
            )

            if result.stderr:                                               # captures error messages
                logging.error(f"Model error: {result.stderr.strip()}")      # logs errors
                return "neutral"                                            # default to neutral in case of model error

            sentiment = result.stdout.strip().lower()                       # sentiment is saved
            if sentiment in self.VALID_SENTIMENTS:                          # valid sentiment
                return sentiment

            logging.warning(f"Unexpected output: {result.stdout.strip()}")  # logs warning
            return "unknown"                                                # handle unexpected outputs

        except subprocess.SubprocessError as e:                             
            logging.error(f"Error in sentiment analysis subprocess: {e}")   # logs errors and returns neutral
            return "neutral"
        except Exception as e:
            logging.error(f"Unexpected error: {e}")                         # logs errors and returns neutral
            return "neutral"

class FileCommentReader(CommentReader):                                 # Implementation for reading comments from a file
    def read_comments(self, input_file):                                # path the text file containing the comments
        """
        Reads comments from a text file, ensuring only valid lines are processed.
        Stops reading after 20 comments to improve efficiency.
        """
        comments = []                                                   # store valid comments
        try:
            with open(input_file, 'r', encoding='utf-8') as file:       # handle various text formates
                for i, line in enumerate(file):
                    if i >= 20:                                         # cap to 20 comments (with permission)
                        break
                    line = line.strip()
                    if line:                                            # ignore empty lines
                        comments.append(line)
            logging.info(f"Read {len(comments)} comments from {input_file}")
            return comments
        except FileNotFoundError:
            logging.error(f"File not found: {input_file}")              # logs error for if file doesn't exist
            return []
        except UnicodeDecodeError as e:
            logging.error(f"Unicode error reading {input_file}: {e}")   # logs error for files encoding that isn't compatible with UTF-8
            return []
        except Exception as e:
            logging.error(f"Error reading {input_file}: {e}")           # logs error for unexpected errors
            return []

class CommentProcessor:                                             # Class for processing comments with a sentiment analyzer
    def __init__(self, comment_reader: CommentReader, sentiment_analyzer: SentimentAnalyzer):
        self.comment_reader = comment_reader                                        # reads comments from file
        self.sentiment_analyzer = sentiment_analyzer                                # evaluates the sentiment of comments

    def process_comments(self, input_file, output_file):
        """
        Processes comments from a file, performs sentiment analysis, and writes results to an output file.
        """
        comments = self.comment_reader.read_comments(input_file)                    # reads comments
        if not comments:
            logging.warning(f"No comments to process in {input_file}.")             # logs warning and returns empty list
            return []

        sentiments = []                                                             # sotre analyzed sentiments for each comment
        try:
            with open(output_file, 'w', encoding='utf-8') as outfile:               # open file with UTF-8 encoding to handle data
                for comment in comments:
                    sentiment = self.sentiment_analyzer.analyze_sentiment(comment)  # analyze each comment
                    sentiments.append(sentiment)
                    outfile.write(sentiment + '\n')                                 # write each comment and do new line
            logging.info(f"Processed {len(comments)} comments from {input_file} to {output_file}")  # logs info of comments as it was successful
        except IOError as e:
            logging.error(f"Error writing to {output_file}: {e}")                   # logs error for writing issues
        return sentiments

class BatchCommentProcessor:                                                        # Class for batch processing of comments
    def __init__(self, input_dir, output_dir, comment_processor: CommentProcessor):
        self.input_dir = input_dir                                                  # path to directory containing input files with comments
        self.output_dir = output_dir                                                # path tp directory where output files (with sentiments) will be saved
        self.comment_processor = comment_processor                                  # class to process each individual file
        self.device_sentiments = {}                                                 # dictionary that maps device na,es to their sentiment results

    def process_all_files(self):
        """
        Processes all .txt files in the input directory and stores sentiment results.
        """
        os.makedirs(self.output_dir, exist_ok=True)                                 # creates the output directory if it doesn't exist                                 

        text_files = [f for f in os.listdir(self.input_dir) if f.endswith(".txt")]  # gathers files in input directory with .txt
        if not text_files:
            logging.error(f"No .txt files found in {self.input_dir}")               # logs errors for no .txt files
            return {}

        for file_name in text_files:
            device_name = file_name.replace("_comments.txt", "")                    # removes _comments.txt from file name
            input_file = os.path.join(self.input_dir, file_name)
            output_file = os.path.join(                                             # constructs full paths for the input and output files
                self.output_dir, file_name.replace("_comments", "_sentiments")  
            )
            logging.info(f"Processing {input_file}...")                             # logs infor for start of process
            sentiments = self.comment_processor.process_comments(input_file, output_file)
            self.device_sentiments[device_name] = sentiments                        # maps the device name to the list of sentiments                  

        return self.device_sentiments

class SentimentPlotter:                                                         # Class for plotting sentiment results
    @staticmethod
    def plot_sentiments(device_sentiments, output_file):
        """
        Plots sentiment distribution for all devices in a single file and increments y-axis by 5.
        
        Args:
        - device_sentiments: dict where keys are device names and values are lists of sentiments.
        - output_file: str, path to save the combined sentiment plot.
        """

        fig, ax = plt.subplots(figsize=(10, 6))                                 # inirialize the figure and axes

        devices = list(device_sentiments.keys())                                # extracts the device names
        sentiment_types = ["positive", "negative", "neutral"]                   # list the types of sentiments
        bar_width = 0.2                                                         # bar width
        x_indices = np.arange(len(devices))                                     # creates evenly spaced indices for the x-axis, one per device

        sentiment_counts = {sentiment: [] for sentiment in sentiment_types}     # counts eacb sentiment type
        for device in devices:
            sentiments = device_sentiments[device]
            sentiment_counts["positive"].append(sentiments.count("positive"))   # counts each positive 
            sentiment_counts["negative"].append(sentiments.count("negative"))   # counts each negative
            sentiment_counts["neutral"].append(sentiments.count("neutral"))     # counts each neutral

        for i, sentiment in enumerate(sentiment_types):                         # plot each sentiment type
            ax.bar(
                x_indices + i * bar_width,                                      # adjusts the position of bars for each sentiment type
                sentiment_counts[sentiment],                                    # sets the heights of the bars for the given sentiment
                width=bar_width,                                                # bar width that's initialized
                label=sentiment,                                                # type of sentiment
                color=["blue", "red", "yellow"][i],                             # distinct colors for each sentiment type
            )

        ax.set_xticks(x_indices + bar_width)                                    # configuring plots
        ax.set_xticklabels(devices)                                             # sets positions and labels for the x-axis ticks to match the devices
        ax.set_ylabel("Count")                                                  # x-axis
        ax.set_xlabel("Devices")                                                # y-axis
        ax.set_title("Sentiment Distribution Across Devices")                   # title for chart
        ax.legend()
        
        max_count = max([max(counts) for counts in sentiment_counts.values()])  # finds the max sentiments count
        ax.set_yticks(range(0, max_count + 6, 5))                               # y-axis increment by 5, ensduring scale is consistent and readable

        os.makedirs(os.path.dirname(output_file), exist_ok=True)                # creates directory if it doesn't exist
        plt.savefig(output_file)                                                # saves the plot to the specficied output file path
        logging.info(f"Saved combined sentiment plot at {output_file}")         # logs info for saved sentiment plots
        plt.close()                                                             # frees memory by closing the plot after saving

if __name__ == "__main__":                                                                  # main execution
    input_dir = "./Comments"                                                                # directory where the input files containing comments are stored
    output_dir = "./Processed"                                                              # directory where the processed sentiment results will be saved
    plots_dir = "./Plots"                                                                   # directory where the sentiment distribution plots will be saved
    model_name = "phi3"                                                                     # name of the local language model used for sentiment analysis

    comment_reader = FileCommentReader()                                                    # read comments from input files
    sentiment_analyzer = LocalLLMSentimentAnalyzer(model_name)                              # initialized with the specified model name, used to analyze the sentiment of comments
    comment_processor = CommentProcessor(comment_reader, sentiment_analyzer)                # process comments and determine their sentiments
    batch_processor = BatchCommentProcessor(input_dir, output_dir, comment_processor)       # batch-process all comment files in the input directory using comment_processor to process each file

    logging.info("Starting batch processing...")                                            # logs info got start of batch processing
    device_sentiments = batch_processor.process_all_files()                                 # process .txt files in the input directory, analyzes the sentiment of comments, saves sentiment results to output directory, and maps device to sentiment results

    logging.info("Generating plots...")                                                     # logs info for start of the plot generation
    SentimentPlotter.plot_sentiments(device_sentiments, plots_dir)                          # plots sentiment distribution for all devices usings the device_sentiments dictionary and saves the combined plot to specified plots_dir

    logging.info("Processing and plotting completed.")                                      # logs info for the completion of the entire processing and plotting workflow
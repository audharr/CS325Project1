import os
from scrapper import scraper  # Import Scraper class from Scrapper.py
from Wao import LocalLLM  # Import LocalLLM class from Wao.py
import matplotlib.pyplot as plt
from collections import Counter

class SentimentProcessor:
    def __init__(self, scraper, llm):
        """
        Initializes the processor with Scraper and LocalLLM instances.
        """
        self.scraper = scraper
        self.llm = llm

    def analyze_comments(self, input_file, output_file):
        """
        Analyzes sentiments of comments in the input file and saves results to the output file.
        """
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if line.strip():  # Ignore empty lines
                    # Prepare the LLM query
                    query = f'Please tell me whether this "{line.strip()}" is positive, negative, or neutral.'
                    
                    # Write the query to a temporary file
                    with open("temp_prompt.txt", 'w') as temp:
                        temp.write(query)
                    
                    # Process the query using the LLM
                    self.llm.process("temp_prompt.txt", "temp_output.txt")
                    
                    # Extract the sentiment from the LLM output
                    with open("temp_output.txt", 'r') as temp_output:
                        sentiment = temp_output.read().strip().split()[-1]  # Assume sentiment is the last word
                        outfile.write(sentiment + '\n')  # Save the sentiment to the output file

    def process_all(self, input_files):
        """
        Processes all input files and generates corresponding sentiment files.
        """
        sentiment_files = []
        for input_file in input_files:
            output_file = input_file.replace('_comments.txt', '_sentiments.txt')
            self.analyze_comments(input_file, output_file)
            sentiment_files.append(output_file)
        return sentiment_files

    def plot_sentiments(self, sentiment_files):
        """
        Plots a bar chart showing sentiment distribution for all devices.
        """
        sentiment_counts = {}

        for file in sentiment_files:
            device_name = os.path.basename(file).replace('_sentiments.txt', '')
            with open(file, 'r') as f:
                sentiments = f.read().splitlines()
            sentiment_counts[device_name] = Counter(sentiments)

        # Plot the data
        for device, counts in sentiment_counts.items():
            labels, values = zip(*counts.items())
            plt.bar(labels, values, label=device)

        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title("Sentiment Analysis per Device")
        plt.legend()
        plt.show()


def main():
    # Paths and initialization
    url_file = 'URL.txt'  # File containing URLs for scraping
    scraper = scraper(url_file)
    llm = LocalLLM(model_name='phi3')  # Adjust model name if different

    # Step 1: Scrape comments
    print("Starting to scrape comments...")
    comment_files = scraper.scrape_all()

    # Step 2: Process comments through the LLM
    processor = SentimentProcessor(scraper, llm)
    print("Analyzing sentiments...")
    sentiment_files = processor.process_all(comment_files)

    # Step 3: Visualize results
    print("Plotting results...")
    processor.plot_sentiments(sentiment_files)


if __name__ == "__main__":
    main()

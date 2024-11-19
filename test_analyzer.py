import os
from scrapper import Scraper
from Wao import LocalLLM
from analyzer import SentimentProcessor

def test_scraper():
    scraper = Scraper('test_urls.txt')  # Create a test URL file
    files = scraper.scrape_all()
    assert len(files) > 0  # Ensure files are created
    for file in files:
        assert os.path.exists(file)

def test_llm():
    llm = LocalLLM('phi3')
    input_file = 'test_input.txt'
    output_file = 'test_output.txt'
    with open(input_file, 'w') as f:
        f.write('Please tell me whether this "Test comment" is positive, negative, or neutral?')
    llm.process(input_file, output_file)
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read().strip()
    assert content in ['positive', 'negative', 'neutral']

def test_processor():
    scraper = Scraper('test_urls.txt')
    llm = LocalLLM('phi3')
    processor = SentimentProcessor(scraper, llm)
    input_files = ['device_comments.txt']  # Replace with test files
    output_files = processor.process_all(input_files)
    for output_file in output_files:
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            sentiments = f.read().splitlines()
        assert all(s in ['positive', 'negative', 'neutral'] for s in sentiments)

def test_plotting():
    processor = SentimentProcessor(None, None)
    sentiment_files = ['device1_sentiments.txt', 'device2_sentiments.txt']  # Replace with test files
    processor.plot_sentiments(sentiment_files)  # Ensure no exceptions are raised

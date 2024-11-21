import os
import pytest
from unittest.mock import patch
from analyzer import FileCommentReader, LocalLLMSentimentAnalyzer, CommentProcessor, BatchCommentProcessor, SentimentPlotter

# Test for FileCommentReader
@pytest.fixture
def create_test_file(tmpdir):
    file_path = tmpdir.join("test_comments.txt")
    with open(file_path, 'w') as f:
        f.write("This is great!\nTerrible experience.\n")
    return str(file_path)

def test_read_comments(create_test_file):
    reader = FileCommentReader()
    comments = reader.read_comments(create_test_file)
    assert comments == ["This is great!", "Terrible experience."]

# Test for LocalLLMSentimentAnalyzer (Mocking subprocess)
@patch("subprocess.run")
def test_analyze_sentiment(mock_subprocess):
    mock_subprocess.return_value.stdout = "positive\n"
    analyzer = LocalLLMSentimentAnalyzer(model_name="phi3")
    sentiment = analyzer.analyze_sentiment("This is amazing!")
    assert sentiment == "positive"

# Test for CommentProcessor
@patch("subprocess.run")
def test_process_comments(mock_subprocess, tmpdir):
    mock_subprocess.return_value.stdout = "positive\n"

    input_file = tmpdir.join("test_comments.txt")
    with open(input_file, 'w') as f:
        f.write("I love this!\nI hate this.\n")

    output_file = tmpdir.join("test_sentiments.txt")

    reader = FileCommentReader()
    analyzer = LocalLLMSentimentAnalyzer(model_name="phi3")
    processor = CommentProcessor(reader, analyzer)

    sentiments = processor.process_comments(str(input_file), str(output_file))

    with open(output_file, 'r') as f:
        output = f.read().splitlines()

    assert output == ["positive", "positive"]
    assert sentiments == ["positive", "positive"]

# Test for BatchCommentProcessor
@patch("subprocess.run")
def test_batch_processing(mock_subprocess, tmpdir):
    mock_subprocess.return_value.stdout = "positive\n"

    input_dir = tmpdir.mkdir("input")
    for i in range(3):
        file_path = input_dir.join(f"device_{i}_comments.txt")
        with open(file_path, 'w') as f:
            f.write("This is a great product.\n")

    output_dir = tmpdir.mkdir("output")

    reader = FileCommentReader()
    analyzer = LocalLLMSentimentAnalyzer(model_name="phi3")
    processor = CommentProcessor(reader, analyzer)
    batch_processor = BatchCommentProcessor(str(input_dir), str(output_dir), processor)

    device_sentiments = batch_processor.process_all_files()

    for i in range(3):
        output_file = output_dir.join(f"device_{i}_sentiments.txt")
        with open(output_file, 'r') as f:
            output = f.read().splitlines()
        assert output == ["positive"]
    assert len(device_sentiments) == 3

# Test for SentimentPlotter
def test_plot_sentiments(tmpdir):
    device_sentiments = {
        "device1": ["positive", "negative", "neutral"],
        "device2": ["positive", "positive", "negative"]
    }

    output_dir = tmpdir.mkdir("plots")
    SentimentPlotter.plot_sentiments(device_sentiments, str(output_dir))

    for device in device_sentiments:
        plot_file = output_dir.join(f"{device}_sentiment_plot.png")
        assert os.path.exists(plot_file)

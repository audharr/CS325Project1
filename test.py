import os                       # interact with the os; creating, deleting, renaming, working inside directory paths, directory exists, and environmemt variables
import pytest                   # testing framework; test functions, utilize assert (directly) and pytest(output)
from unittest.mock import patch # patch is part of unittest.mock library; temporarily replaces parts of code with mock objects during testing, ioslate units of code for testing
# taking classes from analyzer.py and testing the classes and created cases
from analyzer import FileCommentReader, LocalLLMSentimentAnalyzer, CommentProcessor, BatchCommentProcessor, SentimentPlotter

# Test for FileCommentReader, ability to read comments from a file
@pytest.fixture
def create_test_file(tmpdir):                              # create a temporary test file that will hold sample comments
    file_path = tmpdir.join("test_comments.txt")           # file is created and provides a temporary directory
    with open(file_path, 'w') as f:                        # opens the temporary file
        f.write("This is great!\nTerrible experience.\n")  # writes in the test comments into temporary test file
    return str(file_path)

def test_read_comments(create_test_file):
    reader = FileCommentReader()                                   # calls function FileCommentReader from analyzer.py to test
    comments = reader.read_comments(create_test_file)              # read comments to ensure output matches
    assert comments == ["This is great!", "Terrible experience."]


# Test for LocalLLMSentimentAnalyzer (Mocking subprocess), analyze sentiment of a comment 
@patch("subprocess.run")                                        # mocking subprocess.run
def test_analyze_sentiment(mock_subprocess):                    
    mock_subprocess.return_value.stdout = "positive\n"          # always returns positive as the value
    mock_subprocess.return_value.stderr = ""                    # for error
    mock_subprocess.return_value.returncode = 0

    analyzer = LocalLLMSentimentAnalyzer(model_name="phi3")     # calls function LocalLLMSentimentAnalyzer from analyzer.py, using phi3 as the model
    sentiment = analyzer.analyze_sentiment("This is amazing!")  # analyze the comment
    assert sentiment == "positive"                              # ensures the sentiment returned is always positive


# Test for CommentProcessor, combines FileCommentReader and LocalLLMSentimentAnalyzer to process files with comments and be able to save the sentiments to an output file
@patch("subprocess.run")
def test_process_comments(mock_subprocess, tmpdir):
    # Fixing the mock setup to simulate no error
    mock_subprocess.return_value.stdout = "positive\n"                          # the model should return positive
    mock_subprocess.return_value.stderr = ""                                    # no error should occur
    mock_subprocess.return_value.returncode = 0                                 # simulate successful execution

    input_file = tmpdir.join("test_comments.txt")                               # the input file with test comments
    
    with open(input_file, 'w') as f:
        f.write("I love this!\nI hate this.\n")                                 # write test comments into the input file

    output_file = tmpdir.join("test_sentiments.txt")                            # the output file where sentiments will be saved

    reader = FileCommentReader()                                                # initialize the reader
    analyzer = LocalLLMSentimentAnalyzer(model_name="phi3")                     # initialize the sentiment analyzer
    processor = CommentProcessor(reader, analyzer)                              # initialize the processor

    sentiments = processor.process_comments(str(input_file), str(output_file))  # process comments

    with open(output_file, 'r') as f:
        output = f.read().splitlines()                                          # read the output file

    # test the output is positive as the model mock was set to return positive
    assert output == ["positive", "positive"]
    assert sentiments == ["positive", "positive"]
    mock_subprocess.assert_called()                                            # ensure subprocess.run was called


# Test for BatchCommentProcessor, processes multiple files in a directory and writes setiments for each comment
@patch("subprocess.run")
def test_batch_processing(mock_subprocess, tmpdir):
    # Fixing the mock setup to simulate no error
    mock_subprocess.return_value.stdout = "positive\n"                                   # the model should return positive
    mock_subprocess.return_value.stderr = ""                                             # no error should occur
    mock_subprocess.return_value.returncode = 0                                          # simulate successful execution

    input_dir = tmpdir.mkdir("input")                                                    # create a temporary directory for input files
    for i in range(3):
        file_path = input_dir.join(f"device_{i}_comments.txt")                           # create 3 comment files
        with open(file_path, 'w') as f:
            f.write("This is a great product.\n")                                        # add a test comment to each file

    output_dir = tmpdir.mkdir("output")                                                  # create a temporary directory for output files

    reader = FileCommentReader()                                                         # initialize the reader
    analyzer = LocalLLMSentimentAnalyzer(model_name="phi3")                              # initialize the sentiment analyzer
    processor = CommentProcessor(reader, analyzer)                                       # initialize the processor
    batch_processor = BatchCommentProcessor(str(input_dir), str(output_dir), processor)  # batch processor

    device_sentiments = batch_processor.process_all_files()                              # process all files

    for i in range(3):  # Iterate over the 3 devices
        output_file = output_dir.join(f"device_{i}_sentiments.txt")                      # output sentiment file path
        with open(output_file, 'r') as f:
            output = f.read().splitlines()                                               # read the output file
        assert output == ["positive"]                                                    # check the output is "positive" for all devices

    assert len(device_sentiments) == 3                                                   # ensure 3 devices were processed
    mock_subprocess.assert_called()                                                      # ensure subprocess.run was called


# Test for SentimentPlotter, generates plots for sentiment data
def test_plot_sentiments(tmpdir):
    device_sentiments = {                                                  # giving 2 sests of data
        "device1": ["positive", "negative", "neutral"],                    
        "device2": ["positive", "positive", "negative"]
    }

    output_dir = tmpdir.mkdir("plots")                                     # creates a temporary directory called output for the to be generated plots
    SentimentPlotter.plot_sentiments(device_sentiments, str(output_dir))   # calls function SentimentPlotter from anaylzer.py using the 2 sets of data provided and the

    for device in device_sentiments:                                       # every device will be a sentiment plot
        plot_file = output_dir.join(f"{device}_sentiment_plot.png")        # sentiment plot is generated and saved
        assert os.path.exists(plot_file)                                   # confirming that the plot file exists in the created directory
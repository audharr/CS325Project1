# Sentiment Analysis Using Local LLM (phi3)

## Project Overview
This project is designed to analyze sentiments of user comments scraped from a website, leveraging the capabilities of a local LLM (phi3). It processes multiple text files containing product comments, classifies sentiments into positive, negative, or neutral, and visualizes the results in easy-to-read bar charts.

---

## Features and Functionality
- **Automated Sentiment Analysis**: Processes text files using the phi3 LLM model to classify sentiments.
- **Batch Processing**: Handles multiple comment files for various products simultaneously.
- **Visualization**: Generates bar charts to display sentiment distributions using Matplotlib.
- **Object-Oriented Design**: Refactored for modularity and maintainability.
- **Testing**: Includes unit tests using pytest to ensure code reliability.

---

## Project Structure
├── `analyzer.py`         # Main sentiment analysis module
├── `wao.py`              # LLM integration module
├── `test.py`             # Automated test cases for the project
├── Comments/             # Input directory containing comments text files
├── Processed/            # Output directory for processed sentiment files
├── Plots/                # Output directory for generated sentiment plots
├── `requirements.yaml`   # Dependency file for the project
├── `README.md`           # Project documentation


## Requirements
To run this project, make sure your environment satisfies the following requirements:
- **Python 3.8:** or higher
- **ollama**: Required to run the phi3 LLM model
- **matplotlib**: For generating sentiment distribution bar charts
- **pytest**: For running unit tests
- **unittest-mock**: For mocking dependencies in unit tests

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/audharr/CS325Project1/project3_final>
   cd project3_final
   
2. **Install Dependencies: Install the necessary libraries by running the following command:**
   ```bash
   pip install -r requirements.yaml

3. **Prepare Directories: Ensure the following directories are set up as described:**

- **Comments/**: Contains input text files with product comments. Each file should:
Be in .txt format
Contain one comment per line
Follow naming conventions like device1_comments.txt
- **Processed/**: Sentiment classification results will be saved here in .txt format. The format will contain one sentiment per line: positive, negative, or neutral.
Example:
plaintext
Copy code
positive
negative
neutral
Plots/: Sentiment distribution plots will be saved as .png files, following naming conventions like device1_sentiment_plot.png.

4. **Run Sentiment Analysis: After setting up your directories, run the `analyzer.py` script to start the sentiment analysis process:**
    ```bash
   python analyzer.py

5. **Run Unit Tests: To test the functionality of the project, run the automated tests using `pytest`:**
   ```bash
   pytest test.py


##Testing
Automated unit tests are included to verify the correctness of the code. Run the following command to execute the tests:

```bash
pytest test.py

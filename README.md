# **Sentiment Analysis Using Local LLM (phi3)**

## **Project Overview**
This project analyzes sentiments of user comments scraped from a website using the local LLM model **phi3**. It processes multiple text files containing product comments, classifies each sentiment as *positive*, *negative*, or *neutral*, and visualizes the results in bar charts for easy interpretation.

---

## **Features**
- **Automated Sentiment Analysis**: Uses the phi3 LLM model to classify sentiments.
- **Batch Processing**: Handles multiple comment files for various products in a single run.
- **Visualization**: Generates bar charts for sentiment distributions using Matplotlib.
- **Object-Oriented Design**: Modularized and refactored for better maintainability.
- **Testing**: Includes unit tests to verify functionality and reliability.

---

## **Project Structure**
```plaintext
project3_final/
├── analyzer.py         # Main sentiment analysis module
├── wao.py              # LLM integration module
├── test.py             # Automated test cases for the project
├── Comments/           # Input directory containing comment text files
├── Processed/          # Output directory for processed sentiment files
├── Plots/              # Output directory for generated sentiment plots
├── requirements.txt    # Dependency file for the project
├── README.md           # Project documentation
```
## Requirements
**System Requirements**
- **Python 3.8:** or higher
- **ollama**: Required to run the phi3 LLM model
- **matplotlib**: For generating sentiment distribution bar charts
- **pytest**: For running unit tests
- **unittest-mock**: For mocking dependencies in unit tests

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <https://github.com/audharr/CS325Project1/project3_final>
   cd project3_final
   
2. **Install the required libraries:**
   ```bash
   pip install -r requirements.yaml

## Usage

1. **Prepare Directories**
- **Comments/:**
   - Place input text files with product comments here.
   - Each file should:
      - Be in `.txt` format
      - Contain one comment per line
      - Follow naming conventions like `device1_comments.txt`, `device2_comments.txt`, etc
- **Processed/:**
   - Sentiment analysis results will be saved here in text files.
   - Each file will contain one sentiment per line corresponding to the comments:
    ```plaintext
   Example Output:
   plaintext
   positive
   negative
   neutral
   ```
- **Plots/:**
  - Bar charts showing sentiment distributions will be saved here as `.png` files.
  - File naming follow conventions like `device1_sentiment_plot.png`, `device2_sentiment_plot.png`.
  - 
2. **Run Sentiment Analysis**
To process the comments and generates sentiment files and plots:
    ```bash
   python analyzer.py

3. **Run Unit Tests**
To verify functionality, run the automated test cases:
   ```bash
   pytest test.py
   
## Visualization
The generated plots will display grouped bars for sentiment counts (positive, negative, neutral) across all devices. For example:
- **Colors:**
   - Positive: **Blue**
   - Negative: **Red**
   - Neutral: **Yellow**
- **Y-axis:** Increment by 5 for clarity.
- **Example Output:** All devices are displayed in a single plost saved as `sentiment_distribution_all_devices.png` in the `Plots/` directory.

## Example Output
1. **Sentiment File:**
   - For `device1_comments.txt`:
     ```plaintext
     positive
     neutral
     negative
     ```
2. **Plot:**
   - A bar chart showing sentiment distribution for all devices in one image.

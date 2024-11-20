import subprocess
import os
from Wao import LocalLLM  # Importing LocalLLM class from Wao.py

def read_comments_from_file(file_path):
    """
    Reads comments from a text file.
    Each line in the file is treated as a separate comment.
    """
    try:
        print(f"[DEBUG] Reading comments from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]  # Return non-empty lines
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return []


def process_comments(input_file, output_file, llm):
    """
    Processes comments from a single file using the LLM and writes sentiments to an output file.
    """
    comments = read_comments_from_file(input_file)
    if not comments:
        print(f"[DEBUG] No comments found in {input_file}. Skipping...")
        return

    print(f"[DEBUG] Processing {len(comments)} comments from {input_file}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for comment in comments:
            query = f'Please respond using only one word: positive, negative, or neutral. Is this comment "{comment}" positive, negative, or neutral?'
            
            try:
                result = subprocess.run(
                    ["ollama", "run", llm.model_name],
                    input=query,
                    text=True,
                    capture_output=True,
                    check=True
                )
                if result.stdout:
                    sentiment = result.stdout.strip().split()[0].lower()
                    outfile.write(sentiment + '\n')
                    print(f"[DEBUG] Extracted sentiment: {sentiment}")
                else:
                    print(f"[ERROR] No output from model for comment: {comment}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] LLM processing failed for comment: {comment}\n{e}")


def process_comment_files(input_dir="./Comments", output_dir="./Processed", llm=None):
    """
    Processes all comment files in a directory and saves sentiment files to the output directory.
    """
    if llm is None:
        llm = LocalLLM(model_name="phi3")  # Initialize LLM if not provided

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):  # Only process .txt files
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name.replace("_comments", "_sentiments"))
            print(f"[DEBUG] Processing file: {input_file}")
            process_comments(input_file, output_file, llm)


if __name__ == "__main__":
    # Example usage
    input_dir = "./Comments"  # Directory containing comment files
    output_dir = "./Processed"  # Directory to save processed comment files
    llm = LocalLLM(model_name="phi3")  # Initialize the LLM interface

    print("[DEBUG] Starting processing...")
    process_comment_files(input_dir, output_dir, llm)
    print("[DEBUG] Processing completed.")

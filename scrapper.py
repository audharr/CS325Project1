import requests                 #makes HTTP requests to web servers allowing the retrieval of web pages
from bs4 import BeautifulSoup   #navigates, searches, and extracts elements, data, and content from web pages
import time                     #allows for measured execution time to avoid getting blocked or overwhelming a server
import re                       #data validation and or extraction

def sanitize_filename(url):                     #creating a valid filename
    return re.sub(r'[<>:"/\\|?*]', '_', url)    #searching for the provided characters and replaces them with an underscore

def scrape_comments(url):                                           #scrapping comments from URL.txt
    try:                                                            #error handling
        headers = {'User-Agent': 'Mozilla/5.0'}                     #mimic a requests coming from a web browser to avoid getting blocked from a server
        response = requests.get(url, headers=headers, timeout=10)   #add a timeout of 10 seconds
        response.raise_for_status()                                 #raise an error if a status code that is an error returns
        soup = BeautifulSoup(response.text, 'html.parser')          #parses the HTML using beautiful soup

        comments = soup.find_all(class_='pre-white-space')          #find the elements with the name 'pre-white-space' which are the comments for the best buy websites
        
        valid_url = sanitize_filename(url)                          #taking the valid url and saving it as valid_url 
        filename = f"{valid_url}_comments.txt"                      #appending _comments.txt to the new filename 
    
        with open(filename, 'w', encoding='utf-8') as file:         #open/create txt file
            for comment in comments:                                #iterates each comment
                file.write(comment.get_text(strip=True) + '\n')     #write each comment to the file

        print(f"Comments saved to: {filename}")                     #indicates the comments were saved and displays the filename

    except Exception as e:                                          #error handling
        print(f"Error fetching {url}: {e}")                         #if an error occurres then this message will print out

def main():                                             #main function
    with open('URL.txt', 'r') as file:                  #opens the URL.txt file
        urls = file.readlines()                         #puts into a list to read each URL

    for url in urls:                                    #iterating each URL
        url = url.strip()                               #remove any whitespace
        if url:                                         #if the URL is not empty
            print(f"Scraping comments from: {url}")     #confirming that the URL is getting scraped
            scrape_comments(url)                        #calling the scrape_comments function with URL as the argument allowing for the web page to be scraped
            time.sleep(2)                               #sleep for 2 seconds between requests in order to not overwhelm the sever

if __name__ == '__main__':  #make main as short and small as possible
    main()                  #call the main function
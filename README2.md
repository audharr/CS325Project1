# E-commerce Review Scraper

## Project Overview
This project scrapes user reviews from Best Buy for various versions of selected Sony headphones. It automates the process of collecting customer feedback, facilitating analysis of user sentiments across different product models.

## Functionality
- **Automated Scraping**: Retrieves user comments for specified headphone models from Best Buy.
- **Version Organization**: Saves user comments by product model in separate text files.
- **Data Cleaning**: Extracts relevant comment content while discarding unnecessary HTML elements.

## Product Selection
For this project, we focus on the following Sony headphone models:
1. **Sonos Ace** - [Link](https://www.bestbuy.com/site/reviews/sonos-ace-each-black/6580673?variant=A)
2. **Sony Ultimate Wear Wireless Noise Canceling Headphones** - [Link](https://www.bestbuy.com/site/reviews/sony-ult-wear-wireless-noise-canceling-headphones-black/6576179?variant=A)
3. **Sony WH-CH720N Wireless Noise Canceling Headphones** - [Link](https://www.bestbuy.com/site/reviews/sony-whch720n-wireless-noise-canceling-headphones-black/6533162?variant=A)
4. **Sony WH-1000XM4 Wireless Noise Cancelling Over-the-Ear Headphones** - [Link](https://www.bestbuy.com/site/reviews/sony-wh1000xm4-wireless-noise-cancelling-over-the-ear-headphones-black/6408356?variant=A)

## How to Use

### Prerequisites
Make sure you have Python installed on your system. This project requires the following Python packages:
- `beautifulsoup4`
- `requests`
- `time`
- `re`

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/audharr/CS325Project1>
   cd <CS325Project1>

import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter
import pandas as pd

# Work-around for mod security, simulates you being a real user
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}

# Scrape the website's HTML
url = "https://dev3lop.com/advanced-tableau-consulting-services-texas/"
page = requests.get(url,  headers=headers)
soup = BeautifulSoup(page.content, "html.parser")

# Extract the text from the website
text = soup.get_text()

# Print Text to check if everything passed
#print(text)

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Perform part-of-speech tagging on the tokens
tagged_tokens = nltk.pos_tag(tokens)

# Print the tagged tokens
#print(tagged_tokens)

# List of POS tags to include in the analysis
include_pos = ["NN", "VB", "JJ"]

# Filter the tagged tokens to include only the specified POS tags
filtered_tokens = [(token, pos) for token, pos in tagged_tokens if pos in include_pos]

# Print the filtered tokens
#print(filtered_tokens)

# Count filtered tokens
token_counts = Counter(filtered_tokens)

# Print counts
#print(token_counts)

# Create a DataFrame from the token_counts dictionary
df = pd.DataFrame.from_dict(token_counts, orient='index', columns=['Count'])

# Sort the DataFrame by count in descending order
df = df.sort_values(by='Count', ascending=False)

# Print the data table
print(df.head(10).to_string())

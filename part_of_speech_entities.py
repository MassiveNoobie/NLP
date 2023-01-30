# POS stands for part of speech
# This app helps parse 1 web page, finds entities and POS, Values are sent to csv files
#

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
url = "https://dev3lop.com/data-engineering-consulting-services-austin-texas"
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

# Identify named entities in the text
named_entities = nltk.ne_chunk(tagged_tokens)

# Extract the named entities from the chunked text
entities = []
for chunk in named_entities:
    if hasattr(chunk, 'label'):
        entities.append(' '.join(c[0] for c in chunk))

# Print the named entities
#print(entities)

# Count the occurrences of each named entity
entity_counts = Counter(entities)

# Create a data frame from the entity_counts dictionary
df = pd.DataFrame.from_dict(entity_counts, orient='index', columns=['Count'])
df = df.reset_index()
df = df.rename(columns={"index": "Entity"})

# Sort the data frame by count in descending order
df = df.sort_values(by='Count', ascending=False)

# Define the maximum number of rows to display
max_rows = 10

# Export the dataframe to a csv file
df.to_csv("entity_counts.csv", index=False)

# Display only the first max_rows rows
print(df.head(max_rows))


# Print the sorted data frame
#print(df.to_string())
#print(df.apply(lambda x: x.astype(str).str[:40]).to_string)

# Print the tagged tokens
#print(tagged_tokens)

# List of POS tags to include in the analysis
#include_pos = ["NN", "VB", "JJ"]
include_pos = ["NN", "VB", "JJ", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

# Filter the tagged tokens to include only the specified POS tags
filtered_tokens = [(token, pos) for token, pos in tagged_tokens if pos in include_pos]

# Print the filtered tokens
#print(filtered_tokens)

# Count filtered tokens
token_counts = Counter(filtered_tokens)

# Print counts
#print(token_counts)

# Create a DataFrame from the token_counts dictionary
#df = pd.DataFrame.from_dict(token_counts, orient='index', columns=['Count'])

df = pd.DataFrame.from_dict(token_counts, orient='index', columns=['Count'])
df = df.reset_index()
df = df.rename(columns={"index": "Terms_POS"})


#df = pd.DataFrame.from_dict(token_counts, orient='index', columns=['Terms', 'POS', 'guess','Count'])
# Sort the DataFrame by count in descending order
df = df.sort_values(by='Count', ascending=False)

# Print the data table
#print(df.to_string())
print(df.head(max_rows))
df.to_csv("part_of_speech_counts.csv", index=False)

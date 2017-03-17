from __future__ import division
import pandas as pd
import nltk
import csv
import string
from string import punctuation
from string import digits
from nltk.corpus import stopwords
from collections import Counter


# Load in data from Case Master csv export
df = pd.read_csv('C:/Users/nmannheimer/Desktop/DataScience/Text Analytics/CaseMaster.csv')
case_type_df = df.loc[df['Case Type Detail'].isin(['Connections', 'Data Sources'])]
case_type_df = case_type_df.reset_index(drop=True)
length = len(case_type_df)
print length

# Save Descriptions to a single string
descs = ""
for index, row in case_type_df.iterrows():
    descs += row['No Commas']
    print index/length * 100
print 'Text Blob Created'

# Remove those pesky non-ascii characters
printable = set(string.printable)
descs = filter(lambda x: x in printable, descs)
print 'Invalid Characters Removed'

# Remove digits from the string
descs = descs.translate(None, digits)
print 'Digits Removed'

# Remove punctuation from the string
descs = descs.translate(None, punctuation)
print 'Punctuation Removed'

# Make all words upper-case to remove Tableau vs tableau duplication
descs = descs.title()
print 'All Upper Case'

# Create tokens by removing punctuation and creating a list of all words
tokens = nltk.word_tokenize(descs)
print 'Tokens Created'

# Remove stop words like 'the' or 'and'
# We also need to make the stopwords upper-case
# From this point on operations are much faster because we're into higher performance data structures
cachedStopWords = stopwords.words("english")
cachedStopWords = [word.title() for word in cachedStopWords]
tokens = [word for word in tokens if word not in cachedStopWords]
print 'Tokens Cleaned'

# Count the words by occurrences
counts = Counter(tokens)
print 'Counter Complete'

# Save word counts as a csv
with open('C:/Users/nmannheimer/Desktop/DataScience/Text Analytics/DataSourcesandConnections.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in counts.items():
        writer.writerow([key, value])
print 'Completed csv Saved'
print 'Done!'

import json
import pandas

# Read entire file
posts = []
base = "F:\ResearchData\Summarization\Reddit_tifu\\"
with open(base+'tifu_all_tokenized_and_filtered.json', 'r') as fp:
    for line in fp:
        posts.append(json.loads(line))

dfItem = pandas.DataFrame.from_records(posts)
df_long = dfItem.dropna(subset=['tldr'])
df_short = dfItem.dropna(subset=['title'])

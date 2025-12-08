# Amazon → GraphRec Pipeline

# 1. Preprocess Amazon TXT dataset → rating matrix
preprocess_amazon.py

# 2. Build social / trust graph
extract_trust.py

# 3. Expand to format that GraphRec can use
amazon_graphrec.py

import re
from collections import Counter
import matplotlib.pyplot as plt

# Read the text file
with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Use correct Gutenberg markers for this file
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"
start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Warning: Could not find Gutenberg start/end markers. Using full text.")
    clean_text = text
else:
    clean_text = text[start_idx + len(start_marker):end_idx]

# Extract words (lowercase, no punctuation)
words = re.findall(r"\b[a-zA-Z]+\b", clean_text.lower())

# Count word frequencies
counter = Counter(words)
most_common = counter.most_common(50)

if not most_common:
    print("No words found in the text. Exiting.")
    exit(1)

# Calculate mean value of const (freq * rank)
consts = [(freq * (i + 1)) for i, (word, freq) in enumerate(most_common)]
const_avg = sum(consts) / len(consts)

print(f"Mean value of const (based on the 50 most common words): {const_avg:.2f}")

# Prepare data for plotting
terms = [word for word, _ in most_common]
frequencies = [freq for _, freq in most_common]

plt.figure(figsize=(14, 6))
plt.bar(terms, frequencies)
plt.xticks(rotation=45)
plt.title("Zipf's Law: Frequencies of the 50 Most Common Words in 'Pride and Prejudice'")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("zipf_plot.png")
print("Plot saved as zipf_plot.png")
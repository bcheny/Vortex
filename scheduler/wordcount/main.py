import re

word_count = {}

with open('wcs_10gbx.txt', 'r', encoding='utf-8') as file:
    for line in file:
        words = re.findall(r'\b\w+\b', line.lower())
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

for word, count in word_count.items():
    print(f"{word}: {count}")


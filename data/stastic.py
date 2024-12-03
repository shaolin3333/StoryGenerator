import os
# File path to your text file
file_path = "./data/children_stories.txt"
# file_path = "./smalldemo/dataset/train.csv"
file_path = "./data/children_stories.txt_updated"

# Initialize counters
file_size = 0  # Size in bytes
line_count = 0
word_count = 0
char_count = 0
longest_line = ""
longest_line_length = 0
vocab = set()

# Open the file in read mode
with open(file_path, 'r', encoding='utf-8') as file:
    # Get the file size
    file_size = file.tell()
    
    # Read the file line by line
    for line in file:
        line_count += 1
        word_count += len(line.split())  # Count words in the line
        vocab.update(line.strip().split())
        char_count += len(line)  # Count characters in the line
        if len(line) > longest_line_length:
            longest_line = line
            longest_line_length = len(line)

# File size in bytes
file_size = os.path.getsize(file_path)

# Display the results
print(f"File Size: {file_size} bytes")
print(f"Number of Lines: {line_count}")
print(f"Number of Words: {word_count}")
print(f"Vocabulary Size: {len(vocab)}")
print(f"Number of Characters: {char_count}")
print(f"Longest Line Length: {longest_line_length}")
print(f"Longest Line: {longest_line}")

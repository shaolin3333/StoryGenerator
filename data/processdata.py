file_path = "./data/children_stories.txt"  # Replace with your file path
max_words = 128  # Define the maximum allowed line length

# List to store updated lines
updated_lines = []

# Process the file
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        words = line.split()  # Split the line into words
        while len(words) > max_words:
            # Append the first chunk (max_words) to updated lines
            updated_lines.append(" ".join(words[:max_words]) + "\n")
            # Retain the remaining words for further splitting
            words = words[max_words:]
            print("line too long")
        # Append any remaining words as a new line
        updated_lines.append(" ".join(words) + "\n")

# Write the updated lines back to the original file
output_file = file_path + "_updated"
with open(output_file, 'w', encoding='utf-8') as file:
    file.writelines(updated_lines)

print("File updated successfully.")

#!/bin/bash

# Define the output filename
OUTPUT_FILE="combined_output.txt"

# Clear the output file if it already exists
> "$OUTPUT_FILE"

echo "Combining files into $OUTPUT_FILE..."

# Loop through all files in the current directory
for file in *; do
    # Check if it's a regular file and NOT the script or the output file
    if [ -f "$file" ] && [ "$file" != "$OUTPUT_FILE" ] && [ "$file" != "$(basename "$0")" ]; then
        echo "------------------------------------------" >> "$OUTPUT_FILE"
        echo "FILE: $file" >> "$OUTPUT_FILE"
        echo "------------------------------------------" >> "$OUTPUT_FILE"
        
        # Append the content
        cat "$file" >> "$OUTPUT_FILE"
        
        # Add a newline just in case the file doesn't end with one
        echo -e "\n" >> "$OUTPUT_FILE"
    fi
done

echo "Done! All contents saved to $OUTPUT_FILE."

import re

file_path = r'C:\LCC\DATA\templates\index.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add a markdown-to-HTML helper function after the populateList function
markdown_helper = '''
        // --- MARKDOWN TO HTML HELPER ---
        /**
         * Converts simple markdown formatting to HTML
         * @param {string} text - Text with markdown formatting
         * @returns {string} HTML formatted text
         */
        function markdownToHtml(text) {
            if (!text) return '';
            
            // Convert **bold** to <strong>
            text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            
            // Convert *italic* to <em> (but not already processed bold)
            text = text.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
            
            // Convert line breaks to <br>
            text = text.replace(/\n/g, '<br>');
            
            return text;
        }
        // --- END OF MARKDOWN HELPER ---

'''

# Find where to insert the markdown helper (after populateList function ends)
insert_position = content.find('        // --- END OF NEW HELPER FUNCTION ---')
if insert_position != -1:
    insert_position = content.find('\n', insert_position) + 1
    content = content[:insert_position] + markdown_helper + content[insert_position:]

# Now update the populateList function to handle objects properly
old_foreach = '''            items.forEach(item => {
                const li = document.createElement('li');
                // Clean up the start of the string (remove \"*\", \"â€¢\", \"-\", etc.)
                li.textContent = item.replace(/^[\s*â€¢-]+/g, '').trim();
                listElement.appendChild(li);
            });'''

new_foreach = '''            items.forEach(item => {
                const li = document.createElement('li');
                // Clean up the start of the string (remove \"*\", \"â€¢\", \"-\", etc.)
                const cleanText = item.replace(/^[\s*â€¢-]+/g, '').trim();
                // Convert markdown to HTML and set as innerHTML
                li.innerHTML = markdownToHtml(cleanText);
                listElement.appendChild(li);
            });'''

content = content.replace(old_foreach, new_foreach)

# Update the analysis_part1 to use innerHTML with markdown conversion
old_part1 = '''                document.getElementById('analysis_part1').textContent =
                    data.analysis.what_has_happened || \"No information available.\";'''

new_part1 = '''                const part1Element = document.getElementById('analysis_part1');
                const part1Text = data.analysis.what_has_happened || \"No information available.\";
                part1Element.innerHTML = markdownToHtml(part1Text);'''

content = content.replace(old_part1, new_part1)

# Update storySummary and legalSummary to use innerHTML with markdown
old_story = '''                document.getElementById('storySummary').textContent = data.storySummary || \"No summary available.\";'''
new_story = '''                const storyElement = document.getElementById('storySummary');
                storyElement.innerHTML = markdownToHtml(data.storySummary || \"No summary available.\");'''

content = content.replace(old_story, new_story)

old_legal = '''                document.getElementById('legalSummary').textContent = data.legalSummary || \"No summary available.\";'''
new_legal = '''                const legalElement = document.getElementById('legalSummary');
                legalElement.innerHTML = markdownToHtml(data.legalSummary || \"No summary available.\");'''

content = content.replace(old_legal, new_legal)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully updated index.html with markdown rendering!')

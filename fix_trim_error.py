import re

file_path = r'C:\LCC\DATA\templates\index.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Array case
old_array = "                items = textData.filter(item => item && item.trim() !== '');"
new_array = """                items = textData
                    .map(item => typeof item === 'string' ? item : String(item || ''))
                    .filter(item => item.trim() !== '');"""

content = content.replace(old_array, new_array)

# Fix 2: String case (add safety)
old_string = "                items = textData.split('\\n').filter(item => item.trim() !== '');"
new_string = "                items = textData.split('\\n').map(item => String(item)).filter(item => item.trim() !== '');"

content = content.replace(old_string, new_string)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully fixed index.html!')

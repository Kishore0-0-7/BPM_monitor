import os

# Path to the Flask helpers.py file
flask_helpers_path = r"D:\CodeSpace\BPM\.venv\lib\site-packages\flask\helpers.py"

# Read the file
with open(flask_helpers_path, 'r') as f:
    content = f.read()

# Replace the import
if 'from werkzeug.urls import url_quote' in content:
    content = content.replace(
        'from werkzeug.urls import url_quote',
        'from werkzeug.urls import quote as url_quote'
    )
    
    # Write the updated content
    with open(flask_helpers_path, 'w') as f:
        f.write(content)
    
    print("Fixed Flask helpers.py import")
else:
    print("Import statement not found. File might have already been modified.") 
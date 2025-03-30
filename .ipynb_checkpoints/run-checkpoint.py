from app import create_app
import os
from config import Config

# Create the Flask application instance
app = create_app(Config)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Health Assessment Dashboard...")
    print(f"Open http://localhost:{port} in your browser")
    
    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=port)
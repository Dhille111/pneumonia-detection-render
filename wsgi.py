"""
Production WSGI entry point for deployment
Compatible with Gunicorn, uWSGI, and other WSGI servers
"""
from app import app

if __name__ == "__main__":
    # This is for development only
    app.run(debug=False)

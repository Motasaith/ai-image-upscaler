import uvicorn
import multiprocessing
import time
import webbrowser
import http.server
import socketserver
import os
import sys

# CONFIGURATION
# 0.0.0.0 is REQUIRED for Docker. 
# It tells the app to accept connections from outside the container.
HOST = "0.0.0.0" 
API_PORT = 8001
DASHBOARD_PORT = 8091

def run_api():
    """Runs the FastAPI Backend"""
    print(f"🧠 Starting API on http://{HOST}:{API_PORT}")
    uvicorn.run(
        "app.main:app", 
        host=HOST, 
        port=API_PORT, 
        reload=True,
        log_level="info"
    )

def run_dashboard():
    """Runs a simple HTTP server for the Frontend"""
    web_dir = os.path.join(os.path.dirname(__file__), 'app', 'static')
    os.chdir(web_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    class QuietHandler(Handler):
        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer((HOST, DASHBOARD_PORT), QuietHandler) as httpd:
        print(f"🎨 Starting Dashboard on http://{HOST}:{DASHBOARD_PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    api_process = multiprocessing.Process(target=run_api)
    dashboard_process = multiprocessing.Process(target=run_dashboard)

    try:
        api_process.start()
        time.sleep(2)
        dashboard_process.start()

        print("-" * 50)
        print(f"🚀 DOCKER READY!")
        print(f"👉 Open this link in your browser: http://localhost:{DASHBOARD_PORT}")
        print("-" * 50)

        # Note: webbrowser.open() usually doesn't work inside Docker, 
        # so we rely on the print message above.
        
        api_process.join()
        dashboard_process.join()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        api_process.terminate()
        dashboard_process.terminate()
        sys.exit(0)
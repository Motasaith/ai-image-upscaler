import multiprocessing
import time
import http.server
import socketserver
import os
import sys
import subprocess
import uvicorn # Needed for Windows fallback

# CONFIGURATION
HOST = "0.0.0.0" 
API_PORT = 8001
DASHBOARD_PORT = 8091

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

def run_api_windows():
    """Fallback for Windows: Uses Uvicorn directly"""
    print(f"🪟 Windows detected: Running Uvicorn (Dev Mode) on http://{HOST}:{API_PORT}")
    uvicorn.run(
        "app.main:app", 
        host=HOST, 
        port=API_PORT, 
        log_level="info"
    )

def run_api_linux():
    """Production for Linux/Docker: Uses Gunicorn"""
    print(f"🐧 Linux detected: Running Gunicorn (Production Mode) on http://{HOST}:{API_PORT}")
    
    command = [
        "gunicorn",
        "-k", "uvicorn.workers.UvicornWorker",
        "app.main:app",
        "--bind", f"{HOST}:{API_PORT}",
        "--workers", "1",
        "--timeout", "300",
        "--access-logfile", "-" 
    ]
    # Replaces the current process with Gunicorn (saves memory)
    # Note: On Linux/Mac this works. On Windows, this function won't be called.
    subprocess.run(command)

if __name__ == "__main__":
    # 1. Start Dashboard (Always runs in background)
    dashboard_process = multiprocessing.Process(target=run_dashboard)
    dashboard_process.start()
    
    time.sleep(1)
    
    try:
        print("-" * 50)
        print(f"🚀 SERVER STARTING...")
        print(f"👉 Dashboard: http://localhost:{DASHBOARD_PORT}")
        print(f"👉 API:       http://localhost:{API_PORT}")
        print("-" * 50)

        # 2. Start API based on OS
        if sys.platform == "win32":
            # WINDOWS
            run_api_windows()
        else:
            # LINUX / DOCKER
            run_api_linux()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        dashboard_process.terminate()
        sys.exit(0)
"""
dashboard/server.py -- Local web server for the live dashboard.
Run this alongside runner.py to get real-time experiment monitoring.
"""
import os
import sys
import json
import http.server
import socketserver
import threading
import webbrowser
from datetime import datetime

PORT       = 8080
DASH_DIR   = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(DASH_DIR, "state.json")

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASH_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/state.json" or self.path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with open(STATE_FILE) as f:
                    data = f.read()
            except FileNotFoundError:
                data = json.dumps({"status": "waiting", "history": [], "total_exps": 0})
            self.wfile.write(data.encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # suppress access logs

def start_server(open_browser=True):
    handler = DashboardHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"[dashboard] Server running at http://localhost:{PORT}")
        if open_browser:
            threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[dashboard] Server stopped.")

if __name__ == "__main__":
    start_server(open_browser=True)

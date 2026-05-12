#!/usr/bin/env python3
import argparse
import glob
import http.server
import json
import os
import socketserver
import time

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(WEB_DIR, "results")


class Handler(http.server.SimpleHTTPRequestHandler):
    results_path = DEFAULT_RESULTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/api/results", "/api/results/"):
            self.serve_results()
        elif path in ("/api/status", "/api/status/"):
            self.serve_status()
        else:
            super().do_GET()

    def serve_results(self):
        result_file = self.resolve_result_file()
        if not result_file:
            self.send_json({"status": "waiting"})
            return

        try:
            with open(result_file, "r", encoding="utf-8") as f:
                text = f.read()
            json.loads(text)
        except Exception as exc:
            self.send_json({"error": str(exc)}, code=500)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def serve_status(self):
        result_file = self.resolve_result_file()
        exists = bool(result_file)
        mtime = os.path.getmtime(result_file) if result_file else 0
        self.send_json({
            "ready": exists,
            "file": result_file,
            "mtime": mtime,
            "mtime_str": time.ctime(mtime) if exists else None,
        })

    def resolve_result_file(self):
        path = self.results_path
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            return None

        files = glob.glob(os.path.join(path, "results_*.json"))
        files += glob.glob(os.path.join(path, "*", "results_*.json"))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            return
        super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="TSMM benchmark dashboard server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()

    Handler.results_path = os.path.abspath(args.results)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        print(f"TSMM dashboard: http://localhost:{args.port}")
        print(f"Results path: {Handler.results_path}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

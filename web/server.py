#!/usr/bin/env python3
import argparse
import glob
import http.server
import json
import os
import socketserver
import time
import urllib.parse

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(WEB_DIR, "results")


class Handler(http.server.SimpleHTTPRequestHandler):
    results_path = DEFAULT_RESULTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path in ("/api/results", "/api/results/"):
            self.serve_results(query)
        elif path in ("/api/runs", "/api/runs/"):
            self.serve_runs()
        elif path in ("/api/status", "/api/status/"):
            self.serve_status(query)
        else:
            super().do_GET()

    def serve_results(self, query):
        result_file = self.resolve_result_file(query)
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

    def serve_runs(self):
        self.send_json({"runs": self.list_runs()})

    def serve_status(self, query):
        result_file = self.resolve_result_file(query)
        exists = bool(result_file)
        mtime = os.path.getmtime(result_file) if result_file else 0
        self.send_json({
            "ready": exists,
            "file": result_file,
            "mtime": mtime,
            "mtime_str": time.ctime(mtime) if exists else None,
            "runs": self.list_runs(),
        })

    def resolve_result_file(self, query=None):
        query = query or {}
        requested_run = self.first_query_value(query, "run")
        requested_layout = self.first_query_value(query, "layout")
        requested_file = self.first_query_value(query, "file")

        path = self.results_path
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            return None

        if requested_run:
            run_dir = self.safe_run_dir(requested_run)
            if not run_dir:
                return None
            if requested_file:
                base = os.path.basename(requested_file)
                candidate = os.path.abspath(os.path.join(run_dir, base))
                if candidate.startswith(run_dir + os.sep) and os.path.isfile(candidate):
                    return candidate
                return None
            files = self.result_files_in(run_dir, requested_layout)
            return max(files, key=os.path.getmtime) if files else None

        files = self.result_files_in(path, requested_layout)
        for run in self.list_run_dirs():
            files += self.result_files_in(run, requested_layout)
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def list_runs(self):
        runs = []
        for run_dir in self.list_run_dirs():
            files = self.result_files_in(run_dir)
            if not files:
                continue
            mtime = max(os.path.getmtime(f) for f in files)
            layouts = sorted({self.layout_from_file(f) for f in files if self.layout_from_file(f)})
            summaries = {
                name: os.path.isfile(os.path.join(run_dir, name))
                for name in ("gflops.csv", "gflops_summary.json")
            }
            runs.append({
                "id": os.path.basename(run_dir),
                "mtime": mtime,
                "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                "layouts": layouts,
                "files": [os.path.basename(f) for f in sorted(files)],
                "summary": summaries,
            })

        root_files = self.result_files_in(self.results_path)
        if root_files:
            mtime = max(os.path.getmtime(f) for f in root_files)
            runs.append({
                "id": ".",
                "mtime": mtime,
                "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                "layouts": sorted({self.layout_from_file(f) for f in root_files if self.layout_from_file(f)}),
                "files": [os.path.basename(f) for f in sorted(root_files)],
                "summary": {
                    "gflops.csv": os.path.isfile(os.path.join(self.results_path, "gflops.csv")),
                    "gflops_summary.json": os.path.isfile(os.path.join(self.results_path, "gflops_summary.json")),
                },
            })

        runs.sort(key=lambda x: x["mtime"], reverse=True)
        return runs

    def list_run_dirs(self):
        if not os.path.isdir(self.results_path):
            return []
        dirs = []
        for name in os.listdir(self.results_path):
            full = os.path.abspath(os.path.join(self.results_path, name))
            if os.path.isdir(full):
                dirs.append(full)
        return dirs

    def result_files_in(self, directory, layout=None):
        if not os.path.isdir(directory):
            return []
        if layout in ("row", "col"):
            pattern = f"results_{layout}_*.json"
        else:
            pattern = "results_*.json"
        return [f for f in glob.glob(os.path.join(directory, pattern)) if os.path.isfile(f)]

    def safe_run_dir(self, run_id):
        if run_id == ".":
            return os.path.abspath(self.results_path)
        base = os.path.abspath(self.results_path)
        candidate = os.path.abspath(os.path.join(base, os.path.basename(run_id)))
        if not candidate.startswith(base + os.sep):
            return None
        if not os.path.isdir(candidate):
            return None
        return candidate

    @staticmethod
    def first_query_value(query, key):
        value = query.get(key, [""])[0]
        return value.strip()

    @staticmethod
    def layout_from_file(path):
        name = os.path.basename(path)
        if name.startswith("results_row_"):
            return "row"
        if name.startswith("results_col_"):
            return "col"
        return ""

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

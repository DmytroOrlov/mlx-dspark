import http.client
import http.server
import json
import threading
import time
from pathlib import Path

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 13888
LISTEN_PORT = 13885
RUN = Path(__file__).resolve().parent

HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

def redacted(headers):
    out = {}
    for k, v in headers.items():
        out[k] = "<redacted>" if k.lower() == "authorization" else v
    return out

def extract_assistant(raw):
    text = raw.decode("utf-8", "replace")
    pieces = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content")
            if content is None:
                content = message.get("content")
            if isinstance(content, str):
                pieces.append(content)
    if pieces:
        return "".join(pieces)
    try:
        obj = json.loads(text)
        return obj["choices"][0]["message"]["content"]
    except Exception:
        return ""

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        with (RUN / "proxy-access.log").open("a") as f:
            f.write((fmt % args) + "\n")

    def do_GET(self):
        self.forward(False)

    def do_POST(self):
        self.forward(self.path.startswith(("/v1/chat/completions", "/v1/messages")))

    def forward(self, capture):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""

        if capture:
            (RUN / "request.bin").write_bytes(body)
            (RUN / "request-headers.json").write_text(
                json.dumps(redacted(self.headers), indent=2, ensure_ascii=False)
            )
            try:
                obj = json.loads(body)
                (RUN / "request.json").write_text(
                    json.dumps(obj, indent=2, ensure_ascii=False)
                )
            except Exception:
                pass

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in HOP and k.lower() != "host"
        }

        conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=600)
        conn.request(self.command, self.path, body=body, headers=headers)
        resp = conn.getresponse()

        response_headers = {
            k: v for k, v in resp.getheaders()
            if k.lower() not in HOP and k.lower() != "content-length"
        }

        stream = False
        if capture:
            try:
                stream = bool(json.loads(body).get("stream"))
            except Exception:
                pass

        self.send_response(resp.status)
        for k, v in response_headers.items():
            self.send_header(k, v)

        raw = bytearray()

        if stream:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read1(8192)
                    if not chunk:
                        break
                    raw.extend(chunk)
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            data = resp.read()
            raw.extend(data)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        if capture:
            (RUN / "response.bin").write_bytes(raw)
            (RUN / "response-headers.json").write_text(
                json.dumps(dict(resp.getheaders()), indent=2, ensure_ascii=False)
            )
            assistant = extract_assistant(bytes(raw))
            (RUN / "assistant.txt").write_text(assistant)

            try:
                request = json.loads(body)
                summary = {
                    "path": self.path,
                    "stream": request.get("stream"),
                    "model": request.get("model"),
                    "message_count": len(request.get("messages", [])),
                    "messages": request.get("messages", []),
                    "reasoning_effort": request.get("reasoning_effort"),
                    "temperature": request.get("temperature"),
                    "top_p": request.get("top_p"),
                    "top_k": request.get("top_k"),
                    "max_tokens": request.get("max_tokens"),
                    "max_completion_tokens": request.get("max_completion_tokens"),
                    "assistant_first_1000": assistant[:1000],
                }
                (RUN / "summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False)
                )
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            except Exception as e:
                print(f"capture parse error: {e}", flush=True)

            threading.Thread(
                target=lambda: (time.sleep(1), self.server.shutdown()),
                daemon=True,
            ).start()

        conn.close()

server = http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
print(f"CAPTURE_PROXY=http://127.0.0.1:{LISTEN_PORT}/v1", flush=True)
print("Send exactly ONE problematic UI request now.", flush=True)
server.serve_forever()
server.server_close()

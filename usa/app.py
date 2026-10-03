import html
import ipaddress
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "Include", ".env"))

MAX_PAGE_BYTES = 1_000_000
MAX_TEXT_CHARACTERS = 30_000

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Page Reader</title>
  <style>
    :root { color-scheme: light; font-family: Georgia, 'Times New Roman', serif; background: #f1f3ed; color: #202922; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: linear-gradient(135deg, #f1f3ed 0 68%, #e5ece4 68%); }
    main { width: min(760px, 100%); margin: 0 auto; padding: 56px 24px 72px; }
    .eyebrow { color: #39705a; font: 700 12px/1.2 Arial, sans-serif; letter-spacing: 1.5px; text-transform: uppercase; }
    h1 { max-width: 600px; margin: 12px 0 10px; font-size: clamp(38px, 8vw, 62px); font-weight: 500; line-height: 1.02; }
    .intro { margin: 0 0 34px; color: #59665d; font: 16px/1.6 Arial, sans-serif; }
    form { display: grid; gap: 18px; }
    label { display: grid; gap: 8px; font: 700 13px/1.3 Arial, sans-serif; }
    input, textarea { width: 100%; border: 1px solid #c8d1c8; border-radius: 4px; background: #fff; color: #202922; padding: 13px 14px; font: 15px/1.45 Arial, sans-serif; }
    input:focus, textarea:focus { outline: 2px solid #39705a; outline-offset: 2px; }
    textarea { min-height: 118px; resize: vertical; }
    button { justify-self: start; min-height: 46px; border: 0; border-radius: 4px; background: #245942; color: #fff; padding: 0 20px; font: 700 14px Arial, sans-serif; cursor: pointer; }
    button:hover { background: #183f2d; }
    button:disabled { cursor: wait; opacity: .65; }
    #status { min-height: 22px; margin: 20px 0 0; color: #59665d; font: 13px/1.5 Arial, sans-serif; }
    #answer { display: none; margin-top: 14px; border-top: 1px solid #c8d1c8; padding-top: 22px; white-space: pre-wrap; overflow-wrap: anywhere; font: 17px/1.65 Georgia, serif; }
    @media (max-width: 520px) { main { padding: 38px 18px 52px; } }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Read a page with AI</div>
    <h1>What would you like to know?</h1>
    <p class="intro">Enter a public webpage and ask a question about what it says.</p>
    <form id="ask-form">
      <label for="url">Website URL
        <input id="url" name="url" type="url" placeholder="https://example.com/article" required>
      </label>
      <label for="question">Your question
        <textarea id="question" name="question" placeholder="What is this page about?" required></textarea>
      </label>
      <button id="submit" type="submit">Ask about this page</button>
    </form>
    <p id="status" role="status" aria-live="polite"></p>
    <section id="answer" aria-label="Answer"></section>
  </main>
  <script>
    const form = document.querySelector('#ask-form');
    const button = document.querySelector('#submit');
    const status = document.querySelector('#status');
    const answer = document.querySelector('#answer');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.textContent = 'Reading the page and preparing an answer...';
      answer.style.display = 'none';
      try {
        const response = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: form.url.value, question: form.question.value })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'The request could not be completed.');
        answer.textContent = result.answer;
        answer.style.display = 'block';
        status.textContent = `Answer based on ${result.title || result.url}`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>"""


class PageTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in {"br", "p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored_depth:
            text = data.strip()
            if text:
                self.parts.append(text)


def validate_public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a valid public webpage URL starting with http:// or https://.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError) as error:
        raise ValueError("The website address could not be resolved.") from error

    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Only public website addresses can be read.")


class PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        validate_public_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


def fetch_page(url):
    validate_public_url(url)
    request = Request(url, headers={"User-Agent": "PageReader/1.0", "Accept": "text/html"})
    opener = build_opener(PublicRedirectHandler())
    try:
        with opener.open(request, timeout=15) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError("That URL did not return an HTML webpage.")
            content = response.read(MAX_PAGE_BYTES + 1)
            charset = response.headers.get_content_charset() or "utf-8"
            title = response.url
    except HTTPError as error:
        raise ValueError(f"The website returned HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        raise ValueError("The website could not be loaded. Check the URL and try again.") from error

    if len(content) > MAX_PAGE_BYTES:
        raise ValueError("The webpage is too large to read (maximum 1 MB).")

    parser = PageTextParser()
    parser.feed(content.decode(charset, errors="replace"))
    website_text = html.unescape(" ".join(parser.parts))
    if not website_text.strip():
        raise ValueError("No readable text was found on that webpage.")
    return title, website_text[:MAX_TEXT_CHARACTERS]


def answer_question(website_text, question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to usa/Include/.env and restart the app.")

    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user's question using only the website content below.
If the content does not contain enough information to answer, say "I don't have that information in the website."
Do not invent facts.

Website content:
{website_text}

Question:
{question}
""")
    return (prompt | llm).invoke({"website_text": website_text, "question": question}).content


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        page = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10_000:
                raise ValueError("The request is empty or too large.")
            payload = json.loads(self.rfile.read(length))
            url = payload.get("url", "").strip()
            question = payload.get("question", "").strip()
            if not url or not question:
                raise ValueError("Enter both a website URL and a question.")
            title, website_text = fetch_page(url)
            answer = answer_question(website_text, question)
            self.send_json(200, {"url": url, "title": title, "answer": answer})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception:
            self.send_json(502, {"error": "The page or AI service could not complete the request. Check the URL and Groq API key."})

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Page Reader is running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
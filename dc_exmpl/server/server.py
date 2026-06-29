#!/usr/bin/env python3
import http.server
import socketserver

# Переменная нужна для обработки запросов клиента к серверу.
handler = http.server.SimpleHTTPRequestHandler

# Сервер запустится на порте 1234.
with socketserver.TCPServer(("", 1234), handler) as httpd:
    # Сервер будет выполняться постоянно, ожидая запросы
    httpd.serve_forever()
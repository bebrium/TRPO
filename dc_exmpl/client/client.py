#!/usr/bin/env python3
import urllib.request

fp = urllib.request.urlopen("http://localhost:1234/")
# 'decodedContent' соответствует раскодированному ответу сервера
# 'encodedContent' соответствует закодированному ответу сервера
encodedContent = fp.read()
decodedContent = encodedContent.decode("utf-8")

# Выводим содержимое файла, полученного с сервера ('index.html').
print(decodedContent)

# Закрываем соединение с сервером.
fp.close()
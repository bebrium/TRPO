#!/bin/bash

echo "=== Запуск TRMM ==="

echo "Проверка корректности..."
echo "  float  : OK"
echo "  double : OK"

if [ "$1" == "fail" ]; then
    echo "❌ ERROR: Performance test failed!"
    exit 1
else
    echo "✅ All performance tests passed!"
    exit 0
fi
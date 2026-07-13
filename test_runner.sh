#!/bin/bash

echo "=== Запуск реального TRMM бенчмарка ==="


cd OPENBLAS/build
./trmm_benchmark


if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Tests failed!"
    exit 1
fi
Чумаков Алексей ИКС-431

Чтобы запустить тесты склонируйте репозиторий, перейдите в папку OPENBLAS и пропишите:
(1)wsl(необязательно, версия 2)
(2)gcc -O2 CBLAST_test3.c -o cblas_tests -lopenblas -lm
(3)./cblas_tests

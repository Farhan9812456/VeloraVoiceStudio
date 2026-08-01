@echo off
echo ===================================================
echo Starting Zenvyro Labs Advanced Voice Studio...
echo ===================================================
echo.

docker-compose up -d

echo.
echo Waiting for application to initialize...
timeout /t 3 /nobreak > nul

echo Opening browser at http://localhost:7860 ...
start http://localhost:7860

echo.
echo ===================================================
echo Done! App is running in the background.
echo To stop the app, run: docker-compose down
echo ===================================================
echo.
pause

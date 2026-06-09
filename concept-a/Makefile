run:
	cmd /c make kill
	cmd /c make clean
	cmd /c start /b python server/process_paper.py
	@echo Waiting for EasyOCR Engine to initialize...
	cmd /c "FOR /L %i in (1,1,30) DO @(curl -s http://127.0.0.1:5000/ >nul && (echo EasyOCR Is Ready! && goto :out) || (timeout /t 1 /nobreak >nul))" 2>nul & :out
	cmd /c start http://localhost:3000/index.html
	node server/server.js

clean:
	if exist captures\*.png del /Q captures\*.png

pyrun:
	start /b python server/process_paper.py

kill:
	@echo Terminating all running Python backend processes...
	-cmd /c taskkill /F /IM python.exe /T 2>nul
	@echo Terminating any dangling Node instances...
	-cmd /c taskkill /F /IM node.exe /T 2>nul
	@echo System processes cleaned successfully!
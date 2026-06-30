run:
	cmd /c make kill
	cmd /c make clean
	cmd /c make pyrun
	
clean:
	if exist client\logs\* del /Q client\logs\*
	
pyrun:
	start /b python client/launcher.py

kill:
	@echo Terminating all running Python backend processes...
	-cmd /c taskkill /F /IM python.exe /T 2>nul
	@echo System processes cleaned successfully!
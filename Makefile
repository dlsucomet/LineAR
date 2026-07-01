run:
	cmd /c make kill
	cmd /c make clean
	cmd /c make pyrun
	
clean:
	if exist client\logs ( rmdir /S /Q client\logs && mkdir client\logs )
	
pyrun:
	python client\launcher.py

kill:
	@echo Terminating all running Python backend processes...
	-cmd /c taskkill /F /IM python.exe /T 2>nul
	@echo System processes cleaned successfully!
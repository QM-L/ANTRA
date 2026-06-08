set /p user_input=Enter patient dicom name: 
venv\Scripts\python.exe main.py -dicom="%user_input%"
pause
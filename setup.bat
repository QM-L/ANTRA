RMDIR /S /Q venv
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
pause
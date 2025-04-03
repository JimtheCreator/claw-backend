# Start Redis
Start-Process redis-server -WindowStyle Hidden

# Start FastAPI
$env:PYTHONPATH = "$pwd\src;${env:PYTHONPATH}"
uvicorn src.app:app --reload
# Start FastAPI backend in the background
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m", "uvicorn", "backend.api.main:app", "--reload", "--port", "8000"

# Start Next.js frontend in the foreground
cd web
npm run dev

from fastapi import FastAPI
from router import router

app = FastAPI()

# Include all API endpoints from router.py
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

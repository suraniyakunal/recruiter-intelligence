from fastapi import FastAPI
from app.routes.health import router as health_router

app = FastAPI(title="Recruiter Intelligence API")


@app.get("/")
async def root():
    return {"message": "Backend is running"}


app.include_router(health_router)

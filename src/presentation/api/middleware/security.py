from fastapi import Request
from fastapi.middleware import Middleware
from slowapi import Limiter

limiter = Limiter(key_func=lambda: "global", storage_uri="redis://localhost:6379")
middleware = [Middleware(limiter.middleware)]
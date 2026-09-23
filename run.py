"""Entry point: python3 run.py  ->  http://127.0.0.1:8077"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1",
                port=int(os.environ.get("PORT", 8077)), log_level="warning")

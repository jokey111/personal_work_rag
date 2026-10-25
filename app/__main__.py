"""启动入口：uvicorn app.main:app。"""
from dotenv import load_dotenv

load_dotenv()  # 读 .env

from . import config  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

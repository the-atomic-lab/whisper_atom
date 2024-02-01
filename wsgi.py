import uvicorn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a RESTful server.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    uvicorn.run('app.api.server:app', host="0.0.0.0", port=args.port, workers=1)

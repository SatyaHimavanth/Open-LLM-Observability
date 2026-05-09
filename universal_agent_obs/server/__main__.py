"""
python -m universal_agent_obs.server
Starts the collector + trace viewer on port 4317.
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Agent Observability Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", default=4317, type=int, help="Port (default: 4317)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"""
Universal Agent Observability Server
------------------------------------
Dashboard    : http://{args.host}:{args.port}
Ingest       : http://{args.host}:{args.port}/ingest

1. Open the Dashboard in your browser to login (default: admin/password)
2. Obtain your Client ID and Secret from the Settings tab
3. Configure your agent's environment:
     AGENT_OBS_URL=http://localhost:{args.port}
     AGENT_OBS_CLIENT_ID=<your-client-id>
     AGENT_OBS_CLIENT_SECRET=<your-client-secret>
     AGENT_OBS_PROJECT=<your-project-name>
""")

    uvicorn.run(
        "universal_agent_obs.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

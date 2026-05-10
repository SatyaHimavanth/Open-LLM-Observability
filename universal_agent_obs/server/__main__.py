"""
python -m universal_agent_obs.server
Starts the collector + trace viewer on port 4317.
"""

import argparse
import asyncio
import os
import signal
import sys
import threading

import uvicorn


class _SmoothServer(uvicorn.Server):
    def install_signal_handlers(self):
        # The CLI installs its own Ctrl+C handling so the first interrupt can
        # request a clean shutdown and the second can force the process down.
        return


def _install_windows_policy():
    if os.name != "nt":
        return
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass


def _quiet_disconnect_errors(loop):
    default_handler = loop.get_exception_handler()

    def handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
            return
        if default_handler:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def _watch_stdin_for_eof(server: uvicorn.Server):
    def watch():
        try:
            while not server.should_exit:
                char = sys.stdin.read(1)
                if char in ("", "\x04"):
                    print("\n[agent-obs] shutdown requested")
                    server.should_exit = True
                    return
        except Exception:
            return

    if sys.stdin and sys.stdin.isatty():
        threading.Thread(target=watch, daemon=True, name="agent-obs-stdin").start()


async def _serve(args):
    loop = asyncio.get_running_loop()
    _quiet_disconnect_errors(loop)

    config = uvicorn.Config(
        "universal_agent_obs.server.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="warning",
    )
    server = _SmoothServer(config)
    interrupt_count = 0

    def request_shutdown(signum=None, frame=None):
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            print("\n[agent-obs] graceful shutdown requested. Press Ctrl+C again to force.")
            server.should_exit = True
        else:
            print("\n[agent-obs] forcing shutdown")
            server.force_exit = True
            server.should_exit = True

    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, request_shutdown)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, request_shutdown)
        except ValueError:
            pass

    _watch_stdin_for_eof(server)
    await server.serve()


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

    if args.reload:
        uvicorn.run(
            "universal_agent_obs.server.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="warning",
        )
        return

    _install_windows_policy()
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

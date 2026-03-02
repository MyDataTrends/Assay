"""
Main entry point for the Assay MCP Server package.
Allows running via `python -m mcp_server` or `python -m mcp_server run`.
"""

import sys
from mcp_server.server import main

if __name__ == "__main__":
    # If the user called `python -m mcp_server run`, remove the 'run'
    # command so argparse in server.main() doesn't complain.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv.pop(1)
    
    main()

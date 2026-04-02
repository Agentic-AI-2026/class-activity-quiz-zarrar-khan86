from __future__ import annotations

import argparse
import asyncio
import json

from graph import create_workflow


async def _main() -> None:
	parser = argparse.ArgumentParser(description="Planner-Executor agent implemented with LangGraph")
	parser.add_argument(
		"goal",
		nargs="?",
		default="Fetch Q3 sales data and summarize it.",
		help="Goal for the planner-executor workflow",
	)
	parser.add_argument(
		"--servers",
		nargs="*",
		default=["search", "math"],
		help="MCP servers to load (default: search math)",
	)
	args = parser.parse_args()

	app, loaded_tools = await create_workflow(servers=args.servers)

	initial_state = {
		"goal": args.goal,
		"plan": [],
		"current_step": 0,
		"results": [],
	}
	final_state = await app.ainvoke(initial_state)

	print("Loaded tools:")
	print(json.dumps(loaded_tools, indent=2))
	print("\nPlan:")
	print(json.dumps(final_state.get("plan", []), indent=2))
	print("\nResults:")
	print(json.dumps(final_state.get("results", []), indent=2))


if __name__ == "__main__":
	asyncio.run(_main())

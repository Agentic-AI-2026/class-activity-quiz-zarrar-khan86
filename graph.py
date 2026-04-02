from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
	goal: str
	plan: list[dict[str, Any]]
	current_step: int
	results: list[dict[str, Any]]


PLANNER_PROMPT = """You are a planner for a planner-executor agent.
Break the user goal into a minimal, ordered JSON array of steps.

Each step MUST follow this exact schema:
{"step": int, "description": str, "tool": str or null, "args": dict or null}

Rules:
- Use a tool only when needed.
- Use null for tool and args when the step is synthesis/writing.
- Keep arguments concise and directly usable.
- Return only valid JSON (no markdown, no prose).
"""


def _extract_json_array(text: str) -> list[dict[str, Any]]:
	"""Extract and parse a JSON array even if the model wraps it in fences."""
	cleaned = re.sub(r"```(?:json)?", "", text).strip()
	match = re.search(r"\[.*\]", cleaned, re.DOTALL)
	payload = match.group(0) if match else cleaned
	data = json.loads(payload)
	if not isinstance(data, list):
		raise ValueError("Planner did not return a JSON array")
	return data


def _load_llm():
	"""Create an LLM client based on available environment credentials."""
	if os.getenv("OPENAI_API_KEY"):
		from langchain_openai import ChatOpenAI

		return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

	if os.getenv("ANTHROPIC_API_KEY"):
		from langchain_anthropic import ChatAnthropic

		return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), temperature=0)

	if os.getenv("GOOGLE_API_KEY"):
		from langchain_google_genai import ChatGoogleGenerativeAI

		return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"), temperature=0)

	ollama_model = os.getenv("OLLAMA_MODEL")
	if ollama_model:
		from langchain_ollama import ChatOllama

		return ChatOllama(model=ollama_model, temperature=0)

	raise RuntimeError(
		"No LLM provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
		"GOOGLE_API_KEY, or OLLAMA_MODEL."
	)


def _build_mcp_client() -> MultiServerMCPClient:
	root = Path(__file__).resolve().parent
	tools_dir = root / "Tools"

	config: dict[str, dict[str, Any]] = {
		"math": {
			"command": sys.executable,
			"args": [str(tools_dir / "math_server.py")],
			"transport": "stdio",
		},
		"search": {
			"command": sys.executable,
			"args": [str(tools_dir / "search_server.py")],
			"transport": "stdio",
		},
	}

	weather_url = os.getenv("WEATHER_MCP_URL")
	if weather_url:
		config["weather"] = {"url": weather_url, "transport": "streamable_http"}

	return MultiServerMCPClient(config)


async def _load_tools(client: MultiServerMCPClient, servers: list[str]) -> dict[str, Any]:
	tools_map: dict[str, Any] = {}
	for server in servers:
		server_tools = await client.get_tools(server_name=server)
		for tool in server_tools:
			tools_map[tool.name] = tool

	if not tools_map:
		raise RuntimeError("No MCP tools loaded. Check server config and dependencies.")

	return tools_map


def _normalize_tool_args(tool_obj: Any, raw_args: Any) -> dict[str, Any]:
	if not isinstance(raw_args, dict):
		raw_args = {}

	expected = list(getattr(tool_obj, "args", {}).keys())
	if len(expected) == 1 and expected[0] not in raw_args and raw_args:
		return {expected[0]: next(iter(raw_args.values()))}

	return raw_args


def _tool_catalog(tools_map: dict[str, Any]) -> str:
	lines = []
	for name, tool in sorted(tools_map.items()):
		arg_names = ", ".join(getattr(tool, "args", {}).keys()) or "no args"
		lines.append(f"- {name}({arg_names})")
	return "\n".join(lines)


async def create_workflow(servers: list[str] | None = None):
	llm = _load_llm()
	client = _build_mcp_client()
	tools_map = await _load_tools(client, servers or ["search", "math"])

	async def planner_node(state: AgentState) -> AgentState:
		tool_list = _tool_catalog(tools_map)
		response = await llm.ainvoke(
			[
				SystemMessage(content=f"{PLANNER_PROMPT}\n\nAvailable tools:\n{tool_list}"),
				HumanMessage(content=state["goal"]),
			]
		)

		try:
			plan = _extract_json_array(str(response.content))
		except Exception:
			# Safe fallback that still preserves planner -> executor flow.
			plan = [
				{
					"step": 1,
					"description": f"Gather information needed for: {state['goal']}",
					"tool": None,
					"args": None,
				},
				{
					"step": 2,
					"description": "Summarize the gathered information.",
					"tool": None,
					"args": None,
				},
			]

		return {"plan": plan, "current_step": 0, "results": []}

	async def executor_node(state: AgentState) -> AgentState:
		idx = state["current_step"]
		step = state["plan"][idx]
		tool_name = step.get("tool")

		if tool_name and tool_name in tools_map:
			tool_obj = tools_map[tool_name]
			call_args = _normalize_tool_args(tool_obj, step.get("args"))
			step_result = await tool_obj.ainvoke(call_args)
		else:
			context = "\n".join(
				[f"Step {item['step']}: {item['result']}" for item in state["results"]]
			)
			synthesis = await llm.ainvoke(
				[
					SystemMessage(content="Complete the requested step using provided context."),
					HumanMessage(
						content=(
							f"Goal: {state['goal']}\n"
							f"Current step: {step.get('description', '')}\n\n"
							f"Context from previous steps:\n{context}"
						)
					),
				]
			)
			step_result = synthesis.content

		updated_results = state["results"] + [
			{
				"step": step.get("step", idx + 1),
				"description": step.get("description", ""),
				"result": str(step_result),
			}
		]

		return {"results": updated_results, "current_step": idx + 1}

	def continue_or_end(state: AgentState):
		return "executor_node" if state["current_step"] < len(state["plan"]) else END

	graph = StateGraph(AgentState)
	graph.add_node("planner_node", planner_node)
	graph.add_node("executor_node", executor_node)

	graph.add_edge(START, "planner_node")
	graph.add_edge("planner_node", "executor_node")
	graph.add_conditional_edges("executor_node", continue_or_end)

	return graph.compile(), sorted(tools_map.keys())


async def run_goal(goal: str, servers: list[str] | None = None) -> AgentState:
	app, _ = await create_workflow(servers=servers)
	initial_state: AgentState = {
		"goal": goal,
		"plan": [],
		"current_step": 0,
		"results": [],
	}
	final_state = await app.ainvoke(initial_state)
	return final_state

# ─── PLANNER-EXECUTOR AGENT with MCP Tools ────────────────────────────────────

PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
                Each step MUST follow this EXACT schema:
                  {"step": int, "description": str, "tool": str or null, "args": dict or null}
                
                Available MCP tools and their EXACT argument names:
                  - fetch_wikipedia(topic: str)       → look up a topic on Wikipedia
                  - fetch_data_source(source: str)    → source must be one of: sales, customers, expenses
                  - get_weather(city: str)            → get real weather for a city
                
                Use null for tool/args on synthesis or writing steps.
                Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "fetch_wikipedia":  "topic",
    "fetch_data_source": "source",
    "get_weather":      "city",
}

def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Remap hallucinated arg names to the correct parameter."""
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}

async def planner_executor_mcp(goal: str) -> list:
    print(f" Goal: {goal}\n")
    tools, tools_map = await get_mcp_tools(["data", "weather"])
    # planner_llm = ChatAnthropic(model=CHEAP_MODEL, temperature=0)
    planner_llm = llm

    # ── Phase 1: Plan ──────────────────────────────────────────────────────────
    plan_resp = planner_llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=goal)
    ])
    # plan = json.loads(re.sub(r"```json|```", "", plan_resp.content).strip())
    # Extract text safely
    raw_text = plan_resp.content if isinstance(plan_resp.content, str) else plan_resp.content[0].get("text", "")
    
    # Clean and Parse
    clean_json = re.sub(r"```json|```", "", raw_text).strip()
    plan = json.loads(clean_json)
    print(f" Plan ({len(plan)} steps):")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
    print()

    # ── Phase 2: Execute ───────────────────────────────────────────────────────
    results = []
    for step in plan:
        print(f"  Step {step['step']}: {step['description']}")
        tool_name = step.get("tool")

        if tool_name and tool_name in tools_map:
            corrected = safe_args(tool_name, step.get("args") or {})
            result    = await tools_map[tool_name].ainvoke(corrected)
        else:
            # Synthesis step — use LLM with prior results as context
            context  = "\n".join([f"Step {r['step']}: {r['result']}" for r in results])
            response = planner_llm.invoke([
                HumanMessage(content=f"{step['description']}\n\nContext:\n{context}")
            ])
            result = response.content

        print(f"    {str(result)[:150]}\n")
        results.append({"step": step["step"], "description": step["description"], "result": str(result)})

    return results

# ─── Test ─────────────────────────────────────────────────────────────────────
await planner_executor_mcp(
    "Fetch Q3 sales data and look up the history of LangChain on Wikipedia, then summarize both."
)
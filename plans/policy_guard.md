# `policy_guard` node

## Overview
The graph now begins with [`policy_guard`](../app/graph/build_graph.py:51) and conditionally routes to `END` before any planning/search/tooling.

`policy_guard` is implemented in [`app/graph/nodes/policy_guard.py`](../app/graph/nodes/policy_guard.py:1).

## What it does
1. **Prompt-injection hardening**: strips common exfiltration / override phrases (e.g., “ignore previous instructions”, “show system prompt”).
2. **Deterministic high-risk blocking**: regex/keyword heuristics block disallowed categories (poisoning/violence, self-harm methods, weapons/explosives, criminal facilitation, and system prompt exfiltration).
3. **Human-proxy detection**: treats “adult-sized” / 60–120kg disguised targets (e.g., “80 kg chicken”) as a likely human proxy and blocks if the request is about harm.
4. **Optional LLM classifier**: for ambiguous cases, the node can call the LLM to return a small strict JSON object that maps to [`PolicyResult`](../app/graph/policy_models.py:16).

## No-safety-leak guarantees
- Blocked requests **do not** reach planning/search/reasoning/critic/report/judge: routing is handled in [`build_research_graph()`](../app/graph/build_graph.py:51) via conditional edges from `policy_guard`.
- On block, the node clears `state.query` and sets `state.status="blocked"` + `state.final_user_message`.
- Block responses never include system prompts, internal policies, or actionable harmful details.

## How to extend
- Add/adjust deterministic patterns in [`_HARM_TERMS`](../app/graph/nodes/policy_guard.py:31) / [`_INJECTION_PATTERNS`](../app/graph/nodes/policy_guard.py:16).
- Add new categories by extending `PolicyCategory` in [`policy_models.py`](../app/graph/policy_models.py:9) and mapping in `_heuristic_decision()`.
- If you add LLM-side categories, keep the classifier schema in sync with `PolicyCategory`.

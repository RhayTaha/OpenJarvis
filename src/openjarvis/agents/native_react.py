"""NativeReActAgent -- Thought-Action-Observation loop agent.

Renamed from ``ReActAgent`` to clarify this is OpenJarvis's native
implementation, not an integration with an external project.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.agents.prompt_loader import (
    load_few_shot_exemplars,
    load_system_prompt_override,
)
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult, _message_to_dict
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool, build_tool_descriptions

REACT_SYSTEM_PROMPT = """\
You are a ReAct agent. For each step, respond with exactly one of:

1. To think and act:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <json arguments>

2. To give a final answer:
Thought: <your reasoning>
Final Answer: <your answer>

If a user requests one of the tools listed below, issue an Action with its
exact name and JSON object arguments before you answer. Do not claim a tool
was executed unless you received an Observation containing its result.
If no tool call was made, clearly say that you did not use a tool.
Never invent an Observation: only the runtime can supply tool observations.
The only callable tool names are those listed in the available tools section.
Do not add `skill_` or `skill.` to a tool name that does not have it.
If a tool fails, report the failure rather than pretending it succeeded.

# Using Skills

Tools whose names begin with `skill_` are SKILLS. When you call a skill tool,
the response can take one of two forms:

- **Computed result**: The skill ran a deterministic pipeline and returned a
  value (number, string, JSON, etc.). Use the value directly in your answer.

- **Procedural instructions**: The skill returned markdown text describing
  HOW to accomplish a task. Recognize this when the response starts with
  `#` headings, contains bullet lists, or uses phrases like "When asked
  to...", "First...", "Steps:". When you receive instructions:
  1. READ the instructions carefully — they are your playbook
  2. FOLLOW the steps using your OTHER tools (e.g. calculator, web_search,
     shell_exec, file_read) — not the same skill
  3. DO NOT call the same skill again — you already have its instructions
  4. Synthesize a Final Answer from what you learned

{skill_examples}{tool_descriptions}"""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError("Non-JSON numeric constant")


def _parse_json_response(text: str) -> dict[str, str]:
    """Parse a complete JSON envelope without recovering partial tool calls."""
    result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(payload, dict):
            raise ValueError("Expected one JSON object")
        fields = {}
        for key, value in payload.items():
            name = key.strip().lower().replace(" ", "_")
            if name in result:
                if name in fields:
                    raise ValueError("Ambiguous ReAct JSON fields")
                fields[name] = value
        # Ordinary JSON answers without ReAct fields remain plain answers.
        if not fields:
            return result
        for name in ("thought", "action", "final_answer"):
            value = fields.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                result[name] = value.strip()
        if result["action"] and result["final_answer"]:
            raise ValueError("Expected an action or a final answer, not both")
        if result["final_answer"]:
            return result
        if not result["action"]:
            raise ValueError("ReAct JSON needs an action or a final answer")
        arguments = fields.get("action_input", {})
        if isinstance(arguments, str):
            arguments = json.loads(
                arguments,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        if not isinstance(arguments, dict):
            raise ValueError("action_input must be a JSON object")
        result["action_input"] = json.dumps(
            arguments, ensure_ascii=False, allow_nan=False
        )
        return result
    except (ValueError, RecursionError):
        # Do not expose JSON decoder messages or salvage a partial action: the
        # response may contain tool arguments, secrets, or quoted action markers.
        return {
            "thought": "",
            "action": "",
            "action_input": "",
            "final_answer": "",
            "parse_error": (
                "Expected one JSON object with either a nonempty action and "
                "object action_input, or a nonempty final_answer."
            ),
        }



def _resolve_action_name(action: str, available_names: set[str]) -> str:
    """Resolve only a single, known synthetic skill prefix.

    Exact names always win. Unknown actions remain unknown so ToolExecutor
    applies its usual denial; this never adds a new executable capability.
    """
    if action in available_names:
        return action
    for prefix in ("skill.", "skill_"):
        if action.startswith(prefix):
            candidate = action[len(prefix) :]
            if candidate in available_names:
                return candidate
    return action



def _named_tool_contract_guidance(
    requested: set[str], tools: List[BaseTool]
) -> str:
    """Provide exact MCP JSON schemas for explicitly named tools only.

    Generated from advertised ToolSpec rather than trusting the model to
    infer parameter names. No arguments are fabricated or executed here.
    """
    details = []
    for tool in tools:
        spec = tool.spec
        if spec.name not in requested:
            continue
        schema = spec.parameters
        if not isinstance(schema, dict):
            continue
        encoded = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > 6000:
            continue
        details.append(f"Tool {spec.name} input JSON Schema: {encoded}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            fractional = [
                name
                for name, info in properties.items()
                if isinstance(info, dict)
                and info.get("type") in ("number", "integer")
                and isinstance(
                    info.get("maximum", info.get("exclusiveMaximum", 2)),
                    (int, float),
                )
                and info.get("maximum", info.get("exclusiveMaximum", 2)) <= 1
                and isinstance(
                    info.get("minimum", info.get("exclusiveMinimum", -2)),
                    (int, float),
                )
                and info.get("minimum", info.get("exclusiveMinimum", -2)) >= -1
                and any(
                    term in name.lower()
                    for term in ("ratio", "pct", "effectiveness")
                )
            ]
            if fractional:
                details.append(
                    "Fractional percentage fields "
                    + ", ".join(fractional)
                    + " use decimal fractions: 50% = 0.5, 80% = 0.8."
                )
            if spec.name == "analyze_hedge_scenario" and {
                "position_value", "hedge_ratio", "shock_pct",
                "hedge_effectiveness",
            }.issubset(properties):
                details.append(
                    "Hedge semantics: position_value is the position amount; "
                    "hedge_ratio is the protected fraction; hedge_effectiveness "
                    "is the protection efficiency; shock_pct is SIGNED, so "
                    "queda de 10% means shock_pct = -0.10. "
                    "Example only (use the user's actual values): "
                    '{"position_value":10000,"hedge_ratio":0.5,'
                    '"shock_pct":-0.1,"hedge_effectiveness":0.8}.'
                )
    if not details:
        return ""
    return (
        "\n\n# Required tool argument contracts\n"
        + "\n".join(details)
        + "\nUse exact field names and JSON numbers, not prose or percentages. "
        "Do not guess missing parameters. If a tool rejects input, read the "
        "error, correct only supported parameters, and retry at most once per "
        "distinct failed argument set.\n"
    )


@AgentRegistry.register("native_react")
class NativeReActAgent(ToolUsingAgent):
    """ReAct agent: Thought -> Action -> Observation loop."""

    agent_id = "native_react"
    _default_temperature = 0.7
    _default_max_tokens = 1024
    _default_max_turns = 10

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        interactive: bool = False,
        confirm_callback=None,
        skill_few_shot_examples: Optional[List[str]] = None,
        capability_policy: Optional[Any] = None,
        agent_id: Optional[str] = None,
        rate_limiter: Optional[Any] = None,
    ) -> None:
        super().__init__(
            engine,
            model,
            tools=tools,
            bus=bus,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            interactive=interactive,
            confirm_callback=confirm_callback,
            skill_few_shot_examples=skill_few_shot_examples,
            capability_policy=capability_policy,
            agent_id=agent_id,
            rate_limiter=rate_limiter,
        )

    def _parse_response(self, text: str) -> dict:
        """Parse ReAct structured output."""
        stripped = text.strip()
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced and fenced.group(1).lstrip().startswith(("{", "[")):
            return _parse_json_response(fenced.group(1))
        if (
            stripped.startswith("{")
            or re.match(r"\[\s*\{", stripped)
            or stripped.lower().startswith("```json")
            or re.match(r"```[ \t]*\r?\n\s*[\{\[]", stripped)
        ):
            return _parse_json_response(stripped)

        result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(
            r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # Extract Action and Action Input
        action_match = re.search(r"Action:\s*(.+)", text, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\n\n|\nThought:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        # Ground named tool requests in the exact published input schemas.
        available_names = {tool.spec.name for tool in self._tools}
        requested_names = {
            name
            for name in available_names
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", input)
        }
        contract_guidance = _named_tool_contract_guidance(requested_names, self._tools)
        # Build system prompt with rich tool descriptions
        tool_desc = build_tool_descriptions(self._tools)
        # Plan 2B I3: render optimized few-shot skill examples as a section
        # before the tool descriptions. Empty string when not present.
        if self._skill_few_shot_examples:
            skill_examples_block = (
                "## Skill Examples\n\n"
                + "\n\n".join(self._skill_few_shot_examples)
                + "\n\n"
            )
        else:
            skill_examples_block = ""
        # Respect $OPENJARVIS_HOME override for the base template (M2+ work).
        prompt_template = (
            load_system_prompt_override("native_react") or REACT_SYSTEM_PROMPT
        )
        # External overrides may not include the {skill_examples} slot.
        try:
            system_prompt = prompt_template.format(
                tool_descriptions=tool_desc,
                skill_examples=skill_examples_block,
            )
        except KeyError:
            system_prompt = prompt_template.format(tool_descriptions=tool_desc)
            if skill_examples_block:
                system_prompt = system_prompt + "\n\n" + skill_examples_block

        if contract_guidance:
            system_prompt += contract_guidance
        messages = self._build_messages(input, context, system_prompt=system_prompt)

        # Inject few-shot exemplars before the user input
        for ex in load_few_shot_exemplars("native_react"):
            if ex.get("input") and ex.get("output"):
                messages.insert(-1, Message(role=Role.USER, content=ex["input"]))
                messages.insert(-1, Message(role=Role.ASSISTANT, content=ex["output"]))

        # Only enforce tool use when the user explicitly names an installed tool.
        # Do not infer a tool from arbitrary user requests or model-generated text.
        # This prevents a small model from fabricating results when explicitly
        # asked to execute a tool, without forcing tools on ordinary chat.
        retry_count = 0
        failed_calls: set[tuple[str, str]] = set()
        all_tool_results: list[ToolResult] = []
        turns = 0
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _turn in range(self._max_turns):
            turns += 1

            if self._loop_guard:
                messages = self._loop_guard.compress_context(messages)

            result = self._generate(messages)
            usage = result.get("usage", {})
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            content = result.get("content", "")
            parsed = self._parse_response(content)

            if parsed.get("parse_error"):
                self._emit_turn_end(turns=turns)
                return AgentResult(
                    content="Could not parse the model's ReAct response. "
                    + parsed["parse_error"],
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={
                        **total_usage,
                        "response_parse_error": parsed["parse_error"],
                        "messages": [_message_to_dict(m) for m in messages],
                    },
                )

            # The model is not allowed to claim success for a named tool
            # without an actual successful ToolExecutor invocation.
            pending_names = {
                name
                for name in requested_names
                if not any(
                    item.tool_name == name and item.success
                    for item in all_tool_results
                )
            }
            if pending_names and (parsed["final_answer"] or not parsed["action"]):
                if retry_count < 1 and turns < self._max_turns:
                    retry_count += 1
                    # Never feed invented observations back as tool outputs.
                    messages.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "The requested tool has NOT executed successfully. "
                                "Your previous response was not a tool result. "
                                "Issue an Action with an exact available tool name "
                                "and JSON object Action Input. "
                                "Required tool names: "
                                + ", ".join(sorted(pending_names))
                                + ". Do not write an Observation yourself."
                            ),
                        )
                    )
                    continue
                self._emit_turn_end(turns=turns)
                return AgentResult(
                    content=(
                        "Tool execution was not verified for: "
                        + ", ".join(sorted(pending_names))
                        + ". No result is reported."
                    ),
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={
                        **total_usage,
                        "tool_execution_unverified": sorted(pending_names),
                        "messages": [_message_to_dict(m) for m in messages],
                    },
                )

            # Final answer?
            if parsed["final_answer"]:
                self._emit_turn_end(turns=turns)
                msg_dicts = [_message_to_dict(m) for m in messages]
                return AgentResult(
                    content=parsed["final_answer"],
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={**total_usage, "messages": msg_dicts},
                )

            # No action? Treat content as final answer
            if not parsed["action"]:
                self._emit_turn_end(turns=turns)
                msg_dicts = [_message_to_dict(m) for m in messages]
                return AgentResult(
                    content=content,
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={**total_usage, "messages": msg_dicts},
                )

            # Execute action
            messages.append(Message(role=Role.ASSISTANT, content=content))

            # Resolve model-invented skill prefixes only against the exact
            # registered tool names. The executor still applies its ordinary
            # capability checks, confirmation gates, and rate limits.
            action_name = _resolve_action_name(parsed["action"], available_names)
            tool_call = ToolCall(
                id=f"react_{turns}",
                name=action_name,
                arguments=parsed["action_input"] or "{}",
            )

            # Loop guard check before execution
            if self._loop_guard:
                verdict = self._loop_guard.check_call(
                    tool_call.name,
                    tool_call.arguments,
                )
                if verdict.blocked:
                    tool_result = ToolResult(
                        tool_name=tool_call.name,
                        content=f"Loop guard: {verdict.reason}",
                        success=False,
                    )
                    all_tool_results.append(tool_result)
                    observation = f"Observation: {tool_result.content}"
                    messages.append(Message(role=Role.USER, content=observation))
                    continue

            # Repeated invalid JSON arguments cannot improve an outcome.
            # Avoid burning multiple model turns on identical failed calls.
            signature = (tool_call.name, tool_call.arguments)
            if signature in failed_calls:
                self._emit_turn_end(turns=turns)
                return AgentResult(
                    content=(
                        f"Tool {tool_call.name} repeated a failed request. "
                        "No verified result is available. "
                        "Use /tool with the documented JSON arguments."
                    ),
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={
                        **total_usage,
                        "repeated_failed_tool_call": tool_call.name,
                        "messages": [_message_to_dict(m) for m in messages],
                    },
                )
            tool_result = self._executor.execute(tool_call)
            all_tool_results.append(tool_result)
            if not tool_result.success:
                failed_calls.add(signature)
                observation = (
                    "Observation: TOOL EXECUTION FAILED for "
                    + tool_call.name
                    + ". Error: "
                    + str(tool_result.content)
                    + "\nThe preceding call did not produce a verified result. "
                    "Correct the tool arguments using the published schema, "
                    "or report failure. Never invent a successful result."
                )
            else:
                observation = f"Observation: {tool_result.content}"
            messages.append(Message(role=Role.USER, content=observation))

        # A named tool that never succeeded must fail closed even if the
        # model spent all turns proposing invalid actions.
        unverified = {
            name
            for name in requested_names
            if not any(item.tool_name == name and item.success for item in all_tool_results)
        }
        if unverified:
            self._emit_turn_end(turns=turns)
            return AgentResult(
                content="Tool execution was not verified for: "
                + ", ".join(sorted(unverified))
                + ". No result is reported.",
                tool_results=all_tool_results,
                turns=turns,
                metadata={
                    **total_usage,
                    "tool_execution_unverified": sorted(unverified),
                    "messages": [_message_to_dict(m) for m in messages],
                },
            )
        # Max turns exceeded
        msg_dicts = [_message_to_dict(m) for m in messages]
        return self._max_turns_result(
            all_tool_results,
            turns,
            metadata={**total_usage, "messages": msg_dicts},
        )


__all__ = ["NativeReActAgent", "REACT_SYSTEM_PROMPT"]

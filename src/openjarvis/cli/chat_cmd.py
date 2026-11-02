"""``jarvis chat`` — interactive multi-turn chat REPL."""

from __future__ import annotations

import json
import logging
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

from openjarvis.cli._runtime_panel import runtime_cli_options
from openjarvis.cli._tool_names import resolve_tool_names
from openjarvis.cli._voice_chat import VOICE_EXIT, VoiceSession, read_voice_input, speak
from openjarvis.core.config import load_config
from openjarvis.core.events import EventBus
from openjarvis.core.types import Message, Role
from openjarvis.memory import publish_completed_exchange

logger = logging.getLogger(__name__)


def _safe_rich_label(value: object) -> str:
    """Strip terminal controls and escape Rich markup in dynamic labels."""
    from openjarvis.cli._model_switch import sanitize_model_id

    return escape(sanitize_model_id(str(value)))


def _strict_tool_args(raw: str) -> dict:
    """Accept only bounded JSON objects for explicit /tool calls."""
    if len(raw) > 4096:
        raise ValueError("JSON arguments exceed 4096 characters")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON argument")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    args = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    if not isinstance(args, dict):
        raise ValueError("Arguments must be one JSON object")
    return args


def _read_input(prompt: str = "You> ") -> Optional[str]:
    """Read user input with graceful EOF handling."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


@click.command()
@click.option("-e", "--engine", "engine_key", default=None, help="Engine backend.")
@click.option(
    "-m",
    "--model",
    "model_name",
    default=None,
    help=(
        "Model id. Omit or use ``smart`` for preset: [intelligence] model_chat / "
        "model_short / model_long / model_code, then default_model."
    ),
)
@click.option(
    "--pick-model",
    "pick_model",
    is_flag=True,
    default=False,
    help=(
        "Show the model list before chat. Bare ``jarvis`` already does this on a TTY "
        "unless JARVIS_SKIP_MODEL_PICK=1."
    ),
)
@click.option("-a", "--agent", "agent_name", default=None, help="Agent type.")
@click.option("--tools", default=None, help="Comma-separated tool names.")
@click.option("--system", "system_prompt", default=None, help="Custom system prompt.")
@click.option(
    "--persona",
    "persona_name",
    default=None,
    help=(
        "Named persona dir under ~/.openjarvis/personas/<name>/ "
        "(overrides config). Pass 'none' to disable all persona files."
    ),
)
@click.option(
    "--voice",
    "voice_mode",
    is_flag=True,
    default=False,
    help="Enable voice I/O: mic input with silence detection + TTS response playback.",
)
@runtime_cli_options
def chat(
    engine_key: str | None,
    model_name: str | None,
    pick_model: bool,
    agent_name: str | None,
    tools: str | None,
    system_prompt: str | None,
    persona_name: str | None,
    voice_mode: bool,
    num_ctx: int | None,
    num_gpu: int | None,
    skip_runtime_panel: bool,
) -> None:
    """Start an interactive multi-turn chat session.

    Model: omit ``-m`` to use ``[intelligence] model_chat`` if set, else
    ``default_model``; ``-m smart`` is the same. Use ``--pick-model`` to open the
    engine list first (bare ``jarvis`` does this on a TTY unless
    ``JARVIS_SKIP_MODEL_PICK=1``).

    Commands during chat:
      /quit, /exit  — end session
      /clear        — clear conversation history
      /model        — show current model
      /runtime      — Ollama context + GPU offload for this session
      /help         — show available commands
      /history      — show conversation history
      /tools        — list tools available to this agent
      /tool NAME JSON — execute an authorized tool explicitly

    Pass --voice to use microphone input (silence-detection) and hear responses
    read back via text-to-speech (kokoro local or OpenAI TTS).
    """
    console = Console(stderr=True)

    config = load_config()
    bus = EventBus(record_history=False)

    import dataclasses as _dc

    effective_mf = (
        _dc.replace(config.memory_files, persona_name=persona_name)
        if persona_name is not None
        else config.memory_files
    )

    # Resolve engine
    from openjarvis.engine import get_engine
    from openjarvis.intelligence import register_builtin_models

    register_builtin_models()

    resolved = get_engine(config, engine_key)
    if resolved is None:
        console.print("[red]No inference engine available.[/red]")
        sys.exit(1)

    engine_name, engine = resolved
    from openjarvis.cli._model_switch import (
        interactive_pick_model,
        resolve_chat_cli_model,
        tty_wants_model_picker,
    )

    model = ""
    if tty_wants_model_picker(pick_model):
        console.print(
            "[dim]Pick a model below, or press Enter for config default "
            "(intelligence presets / default_model).[/dim]\n",
        )
        picked = interactive_pick_model(console, engine)
        if picked:
            model = picked
    if not model:
        model = resolve_chat_cli_model(
            console=console,
            config=config,
            engine=engine,
            engine_name=engine_name,
            cli_model=model_name,
            chat_variant="chat",
        )
    if not model:
        console.print("[red]No model available.[/red]")
        sys.exit(1)

    from openjarvis.cli._runtime_panel import (
        ChatRuntimeOptions,
        interactive_pick_runtime_options,
        tty_wants_runtime_panel,
    )

    if engine_name != "ollama" and (num_ctx is not None or num_gpu is not None):
        raise click.UsageError(
            "--num-ctx and --num-gpu are supported only with --engine ollama"
        )
    if engine_name == "ollama" and tty_wants_runtime_panel(skip_runtime_panel):
        runtime_opts = interactive_pick_runtime_options(
            console,
            engine_name=engine_name,
            cli_num_ctx=num_ctx,
            cli_num_gpu=num_gpu,
        )
    elif engine_name == "ollama" and (num_ctx is not None or num_gpu is not None):
        runtime_opts = ChatRuntimeOptions(num_ctx=num_ctx, num_gpu=num_gpu)
    else:
        runtime_opts = ChatRuntimeOptions()
    engine_kwargs = runtime_opts.to_engine_kwargs(engine_name=engine_name)

    # Interactive chat is a first-class tool execution path. Apply the same
    # configured engine guardrails, RBAC, rate limiting, and audit bus as ask,
    # serve, and the SDK before constructing any agent.
    from openjarvis.security import setup_security

    security = setup_security(config, engine, bus)
    engine = security.engine

    # Resolve agent (optional)
    # Retain MCP sessions throughout the interactive chat, just as ask does.
    mcp_clients = []
    chat_tools = []
    agent = None
    agent_key = agent_name or config.agent.default_agent
    if agent_key and agent_key != "none":
        try:
            import openjarvis.agents  # noqa: F401 — trigger registration
            from openjarvis.core.registry import AgentRegistry

            if AgentRegistry.contains(agent_key):
                agent_cls = AgentRegistry.get(agent_key)
                kwargs: dict = {"bus": bus}

                if getattr(agent_cls, "accepts_tools", False):
                    tool_names_list = resolve_tool_names(
                        tools,
                        getattr(config.tools, "enabled", None),
                        getattr(config.agent, "tools", None),
                    )
                    if tool_names_list:
                        import openjarvis.tools  # noqa: F401 — trigger registration
                        from openjarvis.core.registry import ToolRegistry
                        from openjarvis.tools._stubs import BaseTool

                        tool_instances = []
                        for tname in tool_names_list:
                            if ToolRegistry.contains(tname):
                                tcls = ToolRegistry.get(tname)
                                if isinstance(tcls, type) and issubclass(
                                    tcls, BaseTool
                                ):
                                    from openjarvis.tools._paths import file_tool_kwargs

                                    tool_instances.append(
                                        tcls(**file_tool_kwargs(tname, config))
                                    )
                                elif isinstance(tcls, BaseTool):
                                    tool_instances.append(tcls)
                    else:
                        tool_instances = []

                    # External MCP tools must be available in interactive chat too.
                    # An explicit --tools argument limits discovery by name.
                    from openjarvis.mcp.loader import load_mcp_tools_from_config

                    mcp_tools, mcp_clients = load_mcp_tools_from_config(
                        config.tools.mcp,
                        allowed_names=set(tool_names_list) if tools is not None else None,
                    )
                    existing = {tool.spec.name for tool in tool_instances}
                    for mcp_tool in mcp_tools:
                        if mcp_tool.spec.name not in existing:
                            tool_instances.append(mcp_tool)
                            existing.add(mcp_tool.spec.name)
                    if tool_instances:
                        chat_tools = tool_instances
                        kwargs["tools"] = tool_instances
                    kwargs["max_turns"] = config.agent.max_turns

                    def _confirm(prompt: str) -> bool:
                        console.print(
                            f"[yellow]Confirm:[/yellow] {prompt} [y/N] ",
                            end="",
                        )
                        ans = input().strip().lower()
                        return ans in ("y", "yes")

                    kwargs["interactive"] = True
                    kwargs["confirm_callback"] = _confirm

                from openjarvis.security.runtime import agent_security_kwargs

                kwargs.update(
                    agent_security_kwargs(
                        agent_cls,
                        capability_policy=security.capability_policy,
                        rate_limiter=security.rate_limiter,
                        agent_id=agent_key,
                    )
                )

                import inspect as _inspect

                if (
                    "prompt_builder"
                    in _inspect.signature(agent_cls.__init__).parameters
                ):
                    from openjarvis.prompt.builder import SystemPromptBuilder

                    kwargs["prompt_builder"] = SystemPromptBuilder(
                        agent_template=config.agent.default_system_prompt or "",
                        memory_files_config=effective_mf,
                        system_prompt_config=config.system_prompt,
                    )

                agent = agent_cls(engine, model, **kwargs)
                from openjarvis.security.runtime import wire_agent_security

                wire_agent_security(
                    agent,
                    bus=bus,
                    capability_policy=security.capability_policy,
                    rate_limiter=security.rate_limiter,
                    agent_id=agent_key,
                )
                if agent is not None and engine_kwargs:
                    # Agents like NativeReActAgent do not accept engine_options
                    # in __init__; session opts live on BaseAgent._engine_options.
                    setattr(agent, "_engine_options", dict(engine_kwargs))
        except Exception as exc:
            console.print(
                f"[yellow]Agent '{_safe_rich_label(agent_key)}' failed: "
                f"{escape(str(exc))}[/yellow]"
            )

    # Keep voice state outside the core chat path so picker/runtime changes can
    # be layered independently. Loaded speech models live for this session.
    voice_session = VoiceSession(config) if voice_mode else None

    # Print banner
    voice_hint = (
        "  [magenta]Voice mode ON[/magenta] — type normally, or press Enter "
        "to speak; silence stops recording.\n"
        if voice_mode
        else ""
    )
    console.print(
        f"[green bold]OpenJarvis Chat[/green bold]\n"
        f"  Engine: [cyan]{_safe_rich_label(engine_name)}[/cyan]  "
        f"Model: [cyan]{_safe_rich_label(model)}[/cyan]"
        f"  Agent: [cyan]{_safe_rich_label(agent_key or 'direct')}[/cyan]\n"
        f"  Runtime: [cyan]{runtime_opts.summary(engine_name=engine_name)}[/cyan]\n"
        f"{voice_hint}"
        f"  Type /help for commands, /tools for tool list, /quit to exit.\n",
    )

    # Background-work status banner (disappears after first user message)
    from openjarvis.cli._bg_state import get_status
    from openjarvis.cli._chat_banner import render_startup_banner

    _banner = render_startup_banner(get_status())
    if _banner:
        console.print(f"[dim cyan]{_banner}[/dim cyan]")

    # Completion-notification dispatcher (fires once per task per session)
    from openjarvis.cli._chat_notifications import NotificationDispatcher

    _notifications = NotificationDispatcher(get_status())

    # Automatic long-term memory — extracts durable facts in the background.
    memory_service = None
    try:
        from openjarvis.memory import build_memory_service

        memory_service = build_memory_service(config, engine, model, event_bus=bus)
        if memory_service is not None:
            memory_service.start()
            console.print("[dim]  Memory: active[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Memory service unavailable: {exc}[/yellow]")
        memory_service = None

    # The document backend and automatic fact store are separate persistence
    # mechanisms. Context injection combines both at read time so facts from
    # previous sessions are immediately available without a manual index step.
    memory_backend = None
    if config.agent.context_from_memory:
        from openjarvis.cli.ask import _get_memory_backend

        memory_backend = _get_memory_backend(config)

    # Conversation state
    if not system_prompt:
        from openjarvis.prompt.builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            agent_template=config.agent.default_system_prompt or "",
            memory_files_config=effective_mf,
            system_prompt_config=config.system_prompt,
        )
        system_prompt = builder.build()

    history: List[Message] = []
    if system_prompt:
        history.append(Message(role=Role.SYSTEM, content=system_prompt))

    # REPL loop
    while True:
        for note in _notifications.diff(get_status()):
            console.print(f"[dim cyan]{note}[/dim cyan]")

        if voice_mode:
            assert voice_session is not None
            result = read_voice_input(console, voice_session)
            if result is VOICE_EXIT:
                console.print("\n[dim]Goodbye![/dim]")
                break
            if result is None:
                continue  # nothing heard, loop again
            user_input = result
        else:
            user_input = _read_input()
            if user_input is None:
                console.print("\n[dim]Goodbye![/dim]")
                break
            user_input = user_input.strip()
            if not user_input:
                continue

        # Handle slash commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break
        elif cmd == "/clear":
            history = []
            if system_prompt:
                history.append(Message(role=Role.SYSTEM, content=system_prompt))
            console.print("[dim]History cleared.[/dim]")
            continue
        elif cmd == "/model":
            console.print(
                f"Model: [cyan]{_safe_rich_label(model)}[/cyan]  "
                f"Engine: [cyan]{_safe_rich_label(engine_name)}[/cyan]"
            )
            continue
        elif cmd == "/runtime":
            console.print(
                f"Runtime: [cyan]{runtime_opts.summary(engine_name=engine_name)}[/cyan]"
            )
            if engine_kwargs:
                console.print(f"  engine kwargs: {engine_kwargs}")
            continue
        elif cmd == "/help":
            console.print(
                "[bold]Commands:[/bold]\n"
                "  /quit, /exit  — end session\n"
                "  /clear        — clear conversation\n"
                "  /model        — show model info\n"
                "  /runtime      — Ollama context + GPU offload for this session\n"
                "  /history      — show conversation\n"
                "  /tools        — show available agent tools\n"
                "  /tool NAME JSON — execute tool with JSON object arguments\n"
                "  /help         — this message"
            )
            continue
        elif cmd == "/tools":
            if not chat_tools or agent is None:
                console.print("[yellow]No tools available to this chat agent.[/yellow]")
            else:
                for tool in chat_tools:
                    console.print(f"  [cyan]{_safe_rich_label(tool.spec.name)}[/cyan]")
            continue
        elif cmd == "/tool" or cmd.startswith("/tool "):
            # Explicit dispatch uses the same security-aware executor as agent actions.
            parts = user_input.split(maxsplit=2)
            if agent is None or not hasattr(agent, "_executor"):
                console.print("[red]No tool-capable agent available.[/red]")
                continue
            if len(parts) != 3:
                console.print("[yellow]Usage: /tool NAME {\"argument\":\"value\"}[/yellow]")
                continue
            name = parts[1]
            if name not in {tool.spec.name for tool in chat_tools}:
                console.print("[red]Unknown or unavailable tool.[/red]")
                continue
            try:
                arguments = _strict_tool_args(parts[2])
            except (ValueError, json.JSONDecodeError) as exc:
                console.print(f"[red]Invalid tool arguments: {escape(str(exc))}[/red]")
                continue
            from openjarvis.core.types import ToolCall

            from uuid import uuid4

            result = agent._executor.execute(
                ToolCall(
                    id=f"cli_tool_{uuid4().hex}",
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False),
                )
            )
            state = "OK" if result.success else "FAILED"
            console.print(
                f"[bold]Tool execution: {_safe_rich_label(name)} — {state}[/bold]"
            )
            console.print(escape(str(result.content)))
            continue
        elif cmd == "/history":
            if not history:
                console.print("[dim]No history yet.[/dim]")
            else:
                for msg in history:
                    role_str = msg.role if isinstance(msg.role, str) else msg.role.value
                    role = role_str.upper()
                    console.print(f"[bold]{role}:[/bold] {msg.content[:200]}")
            continue

        # Add user message
        history.append(Message(role=Role.USER, content=user_input))

        generation_history = history
        agent_context_message = None
        if config.agent.context_from_memory:
            try:
                from openjarvis.memory import load_configured_facts
                from openjarvis.tools.storage.context import (
                    ContextConfig,
                    inject_context,
                )

                if memory_service is not None and hasattr(memory_service, "list_facts"):
                    facts = memory_service.list_facts()
                else:
                    facts = load_configured_facts(config)
                ctx_cfg = ContextConfig(
                    top_k=config.memory.context_top_k,
                    min_score=config.memory.context_min_score,
                    max_context_tokens=config.memory.context_max_tokens,
                )
                context_messages = inject_context(
                    user_input,
                    [] if agent is not None else history,
                    memory_backend,
                    config=ctx_cfg,
                    facts=facts,
                )
                if agent is not None:
                    if context_messages:
                        agent_context_message = context_messages[0]
                else:
                    generation_history = context_messages
            except Exception:
                logger.debug("Failed to inject memory context", exc_info=True)

        # Generate response even when optional memory context is unavailable.
        try:
            if agent is not None:
                from openjarvis.agents._stubs import AgentContext

                agent_context = AgentContext()
                if agent_context_message is not None:
                    agent_context.conversation.add(agent_context_message)
                for msg in history[:-1]:
                    if msg.role != Role.SYSTEM:
                        agent_context.conversation.add(msg)
                response = agent.run(user_input, context=agent_context)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                for used_tool in getattr(response, "tool_results", []) or []:
                    outcome = "OK" if used_tool.success else "FAILED"
                    console.print(
                        f"[dim]Tool executed: "
                        f"{_safe_rich_label(used_tool.tool_name)} — {outcome}[/dim]"
                    )
            else:
                result = engine.generate(
                    generation_history,
                    model=model,
                    **engine_kwargs,
                )
                content = (
                    result.get("content", "")
                    if isinstance(result, dict)
                    else str(result)
                )

            history.append(Message(role=Role.ASSISTANT, content=content))
            console.print()
            console.print(Markdown(content))
            console.print()
            if voice_mode:
                assert voice_session is not None
                speak(content, console, voice_session)

            publish_completed_exchange(
                bus,
                user_input,
                content,
                source="cli.chat",
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Generation interrupted.[/dim]")
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]\n")

    if memory_service is not None:
        memory_service.stop()
    for mcp_client in mcp_clients:
        try:
            mcp_client.close()
        except Exception:
            logger.warning("Failed to close MCP client", exc_info=True)


__all__ = ["chat"]

// Calls `uvx jes hook` on stdin. uvx ships with uv, so there is no separate jes install.
// before_agent_run blocks the user prompt. before_tool_call blocks the call.
// tool_result_persist puts onward in content. message_sending rewrites the
// reply that is about to be delivered.
import { sessionOf, spawnJes, type JesRunner } from "./jes-runner.ts";

type HookContext = { sessionId?: string; sessionKey?: string };

export default function register(
  api: {
    on: (
      name: string,
      handler: (event: Record<string, unknown>, ctx: HookContext) => unknown,
    ) => void;
  },
  run: JesRunner = spawnJes,
) {
  api.on("before_agent_run", (event, ctx) => {
    const prompt = typeof event.prompt === "string" ? event.prompt : "";
    const checked = run({
      stage: "input",
      text: prompt,
      session_id: sessionOf(ctx.sessionId ?? ctx.sessionKey, "openclaw"),
    });
    if (!checked.ok) {
      return { outcome: "block", reason: checked.onward, message: checked.onward };
    }
    return undefined;
  });

  api.on("before_tool_call", (event, ctx) => {
    const tool = typeof event.toolName === "string" ? event.toolName : "";
    const checked = run({
      stage: "tool_call",
      tool,
      arguments: event.params ?? {},
      session_id: sessionOf(ctx.sessionId ?? ctx.sessionKey, "openclaw"),
    });
    if (!checked.ok) {
      return { block: true, blockReason: checked.onward };
    }
    return undefined;
  });

  api.on("tool_result_persist", (event, ctx) => {
    const tool = typeof event.toolName === "string" ? event.toolName : "tool";
    const content =
      typeof event.content === "string" ? event.content : JSON.stringify(event.content ?? "");
    const checked = run({
      stage: "tool_result",
      tool,
      text: content,
      session_id: sessionOf(ctx.sessionId ?? ctx.sessionKey, "openclaw"),
    });
    if (!checked.ok) {
      return { content: checked.onward };
    }
    return undefined;
  });

  api.on("message_sending", (event, ctx) => {
    const text =
      typeof event.content === "string" ? event.content : JSON.stringify(event.content ?? "");
    const checked = run({
      stage: "output",
      text,
      session_id: sessionOf(ctx.sessionId ?? ctx.sessionKey, "openclaw"),
    });
    if (!checked.ok) {
      return { content: checked.onward };
    }
    return undefined;
  });
}

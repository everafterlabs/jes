// Checks each step with jes through ./jes-runner.ts, which `jes runner-settings` prints.
// before_agent_start checks the user prompt and injects the refusal when it
// is blocked. tool_call returns block: true. tool_result replaces content.
import { sessionOf, spawnJes, type JesRunner } from "./jes-runner.ts";

type PiContext = { sessionId?: string };

export default function jesExtension(
  pi: {
    on: (
      name: string,
      handler: (event: Record<string, unknown>, ctx: PiContext) => unknown,
    ) => void;
  },
  run: JesRunner = spawnJes,
) {
  pi.on("before_agent_start", async (event, ctx) => {
    const prompt = typeof event.prompt === "string" ? event.prompt : "";
    const checked = run({
      stage: "input",
      text: prompt,
      session_id: sessionOf(ctx.sessionId, "pi"),
    });
    if (!checked.ok) {
      return {
        message: { customType: "jes", content: checked.onward, display: true },
      };
    }
    return undefined;
  });

  pi.on("tool_call", async (event, ctx) => {
    const tool = typeof event.toolName === "string" ? event.toolName : "";
    const checked = run({
      stage: "tool_call",
      tool,
      arguments: event.input ?? {},
      session_id: sessionOf(ctx.sessionId, "pi"),
    });
    if (!checked.ok) {
      return { block: true, reason: checked.onward };
    }
    return undefined;
  });

  pi.on("tool_result", async (event, ctx) => {
    const tool = typeof event.toolName === "string" ? event.toolName : "tool";
    const text =
      typeof event.content === "string" ? event.content : JSON.stringify(event.content ?? "");
    const checked = run({
      stage: "tool_result",
      tool,
      text,
      session_id: sessionOf(ctx.sessionId, "pi"),
    });
    if (!checked.ok) {
      if (Array.isArray(event.content)) {
        return { content: [{ type: "text", text: checked.onward }] };
      }
      return { content: checked.onward };
    }
    return undefined;
  });
}

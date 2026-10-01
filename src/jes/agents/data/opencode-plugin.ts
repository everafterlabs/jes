// Checks each step with jes through ./jes-runner.ts, which `jes runner-settings` prints.
// OpenCode has no display-only reply hook, so this plugin does not rewrite
// the assistant reply. It checks the user prompt, the tool call, and the
// tool result. A blocked tool result replaces output.output with onward.
import { sessionOf, spawnJes, type JesRunner } from "./jes-runner.ts";

type MessageInput = { sessionID?: string };
type MessageOutput = { message?: { content?: unknown } };
type ToolInput = { tool: string; sessionID?: string };
type ToolArgs = { args?: Record<string, unknown> };
type ToolOutput = { output?: unknown };

export function createJesGuard(run: JesRunner = spawnJes) {
  return async () => {
    return {
      "chat.message": async (input: MessageInput, output: MessageOutput) => {
        const content = output.message?.content;
        const text = typeof content === "string" ? content : JSON.stringify(content ?? "");
        const checked = run({
          stage: "input",
          text,
          session_id: sessionOf(input.sessionID, "opencode"),
        });
        if (!checked.ok) {
          throw new Error(checked.onward);
        }
      },
      "tool.execute.before": async (input: ToolInput, output: ToolArgs) => {
        const checked = run({
          stage: "tool_call",
          tool: input.tool,
          arguments: output.args ?? {},
          session_id: sessionOf(input.sessionID, "opencode"),
        });
        if (!checked.ok) {
          throw new Error(checked.onward);
        }
      },
      "tool.execute.after": async (input: ToolInput, output: ToolOutput) => {
        const text =
          typeof output.output === "string" ? output.output : JSON.stringify(output.output ?? "");
        const checked = run({
          stage: "tool_result",
          tool: input.tool,
          text,
          session_id: sessionOf(input.sessionID, "opencode"),
        });
        if (!checked.ok) {
          output.output = checked.onward;
        }
      },
    };
  };
}

export const JesGuard = createJesGuard();

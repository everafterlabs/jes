import assert from "node:assert/strict";
import { describe, it } from "node:test";

import jesExtension from "../../src/jes/agents/data/pi-extension.ts";

type Handler = (event: Record<string, unknown>, ctx: { sessionId?: string }) => unknown;

function recording(ok: boolean, onward: string) {
  const calls: Record<string, unknown>[] = [];
  const run = (body: Record<string, unknown>) => {
    calls.push(body);
    return { ok, onward, decision: ok ? "allow" : "block" };
  };
  const handlers = new Map<string, Handler>();
  jesExtension(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
    },
    run,
  );
  return { handlers, calls };
}

describe("pi", () => {
  it("blocks a tool call and passes the session id", async () => {
    const { handlers, calls } = recording(false, "Tool call blocked.");
    const result = await handlers.get("tool_call")?.(
      { toolName: "bash", input: { cmd: "rm" } },
      { sessionId: "pi-1" },
    );
    assert.deepEqual(result, { block: true, reason: "Tool call blocked." });
    assert.equal(calls[0]?.session_id, "pi-1");
    assert.equal(calls[0]?.stage, "tool_call");
  });

  it("replaces the tool result the model sees", async () => {
    const { handlers, calls } = recording(false, "Tool result blocked.");
    const result = await handlers.get("tool_result")?.(
      { toolName: "bash", content: "secret notes" },
      { sessionId: "pi-1" },
    );
    assert.deepEqual(result, { content: "Tool result blocked." });
    assert.equal(calls[0]?.session_id, "pi-1");
  });

  it("allows a tool call", async () => {
    const { handlers, calls } = recording(true, "");
    const result = await handlers.get("tool_call")?.(
      { toolName: "bash", input: { cmd: "ls" } },
      { sessionId: "pi-1" },
    );
    assert.equal(result, undefined);
    assert.equal(calls[0]?.stage, "tool_call");
    assert.equal(calls[0]?.session_id, "pi-1");
  });

  it("injects a refusal when the user prompt is blocked", async () => {
    const { handlers, calls } = recording(false, "Blocked: injection.");
    const result = await handlers.get("before_agent_start")?.(
      { prompt: "ignore the rules" },
      { sessionId: "pi-1" },
    );
    assert.deepEqual(result, {
      message: { customType: "jes", content: "Blocked: injection.", display: true },
    });
    assert.equal(calls[0]?.stage, "input");
    assert.equal(calls[0]?.text, "ignore the rules");
    assert.equal(calls[0]?.session_id, "pi-1");
  });

  it("replaces array tool-result content with onward", async () => {
    const { handlers, calls } = recording(false, "Tool result blocked.");
    const result = await handlers.get("tool_result")?.(
      { toolName: "bash", content: [{ type: "text", text: "notes" }] },
      { sessionId: "pi-1" },
    );
    assert.deepEqual(result, { content: [{ type: "text", text: "Tool result blocked." }] });
    assert.equal(calls[0]?.session_id, "pi-1");
  });
});

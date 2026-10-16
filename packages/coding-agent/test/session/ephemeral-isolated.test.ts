/**
 * Evidence-only rendering for side turns: `runEphemeralTurn({ isolated: true })`
 * must send only the frozen supplied prompt — never the session transcript, its
 * system prompt or its tool catalog — and must reject options that would
 * reintroduce session state.
 */
import { afterEach, describe, expect, test, vi } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const SENTINEL = "SESSION_SENTINEL_DO_NOT_SEND";

interface Harness {
	session: AgentSession;
	model: ReturnType<typeof createMockModel>;
	auth: AuthStorage;
	close(): Promise<void>;
}

function buildHarness(fullPayload: string): Harness {
	const model = createMockModel({ responses: [{ content: [fullPayload] }] });
	const auth = createInMemoryAuthStorage();
	const agent = new Agent({
		getApiKey: () => "fixture-key",
		streamFn: model.stream,
		initialState: {
			model,
			systemPrompt: [SENTINEL],
			tools: [],
			messages: [{ role: "user", content: SENTINEL, timestamp: 1 }],
		},
	});
	const session = new AgentSession({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated(),
		modelRegistry: new ModelRegistry(auth, "/nonexistent/fixture-models.yml"),
		sideStreamFn: model.stream,
	});
	return {
		session,
		model,
		auth,
		close: async () => {
			await session.dispose();
			auth.close();
		},
	};
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe("isolated ephemeral rendering", () => {
	test("isolated payload rendering excludes the actual session transcript", async () => {
		const fullPayload = JSON.stringify({ case: "done", reason: "fixture completion evidence" });
		const harness = buildHarness(fullPayload);
		try {
			const reply = await harness.session.runEphemeralTurn({
				promptText: "fixture completion evidence",
				tools: false,
				isolated: true,
				dedupeReply: false,
				signal: AbortSignal.timeout(1000),
			});
			expect(reply.replyText).toBe(fullPayload);
			expect(harness.model.calls.map(call => JSON.stringify(call.context)).join("\n")).not.toContain(SENTINEL);
			expect(harness.model.calls[0]!.context.tools ?? []).toEqual([]);
			expect(JSON.stringify(harness.model.calls[0]!.context.messages)).toContain("fixture completion evidence");
		} finally {
			await harness.close();
		}
	});

	test("isolated rendering sends the fixed evidence-only system block, not the session prompt", async () => {
		const harness = buildHarness("fixture reply");
		try {
			await harness.session.runEphemeralTurn({
				promptText: "fixture completion evidence",
				tools: false,
				isolated: true,
				signal: AbortSignal.timeout(1000),
			});
			const system = harness.model.calls[0]!.context.systemPrompt ?? [];
			expect(system.join("\n")).toContain("Render only the supplied frozen evidence");
			expect(system.join("\n")).not.toContain(SENTINEL);
		} finally {
			await harness.close();
		}
	});

	test("isolated rendering requires tools:false and rejects history and conversationKey", async () => {
		const harness = buildHarness("fixture reply");
		try {
			await expect(
				harness.session.runEphemeralTurn({ promptText: "x", isolated: true, signal: AbortSignal.timeout(1000) }),
			).rejects.toThrow("tools:false");
			await expect(
				harness.session.runEphemeralTurn({
					promptText: "x",
					tools: false,
					isolated: true,
					history: [{ role: "user", content: "prior", timestamp: 1 }],
					signal: AbortSignal.timeout(1000),
				}),
			).rejects.toThrow("history and conversationKey");
			await expect(
				harness.session.runEphemeralTurn({
					promptText: "x",
					tools: false,
					isolated: true,
					conversationKey: "topic",
					signal: AbortSignal.timeout(1000),
				}),
			).rejects.toThrow("history and conversationKey");
			expect(harness.model.calls).toHaveLength(0);
		} finally {
			await harness.close();
		}
	});

	test("isolated rendering enforces maxContextBytes before inference", async () => {
		const harness = buildHarness("fixture reply");
		try {
			await expect(
				harness.session.runEphemeralTurn({
					promptText: "an oversized isolated payload that no small byte budget can carry",
					tools: false,
					isolated: true,
					maxContextBytes: 32,
					signal: AbortSignal.timeout(1000),
				}),
			).rejects.toThrow("exceeds the configured");
			expect(harness.model.calls).toHaveLength(0);
		} finally {
			await harness.close();
		}
	});

	test("a default side turn still carries the session transcript", async () => {
		const harness = buildHarness("fixture reply");
		try {
			await harness.session.runEphemeralTurn({ promptText: "ordinary side turn", signal: AbortSignal.timeout(1000) });
			expect(JSON.stringify(harness.model.calls[0]!.context.messages)).toContain(SENTINEL);
		} finally {
			await harness.close();
		}
	});
});

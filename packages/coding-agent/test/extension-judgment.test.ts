/**
 * Session-owned extension judgment bridge: session binding, identity, immutable
 * requests, the total deadline, cancellation, and evidence-only rendering.
 *
 * Every case drives the real `before_agent_start` handler's `ctx.judge` and the
 * real session journal through the production runner and registry — none of
 * them re-asserts a mock's canned envelope.
 */
import { afterEach, describe, expect, test, vi } from "bun:test";
import * as path from "node:path";
import type { ChoiceQuestion, Model, Questions } from "@oh-my-pi/pi-ai";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import type { ExtensionFactory } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/types";
import type { JudgmentRoute } from "@oh-my-pi/pi-coding-agent/judgment";
import { resolveJudge } from "@oh-my-pi/pi-coding-agent/judgment";
import { initializeExtensions } from "@oh-my-pi/pi-coding-agent/modes/runtime-init";
import { type CreateAgentSessionOptions, createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import {
	captureJson,
	type ExtensionJudgment,
	runExtensionJudgment,
	settleWithin,
} from "@oh-my-pi/pi-coding-agent/session/extension-judgment";
import type { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";
import { asGlobalFetch } from "./helpers/fetch-mock";

import { getBundledModel } from "@oh-my-pi/pi-catalog/models";

const DECISIONS = getBundledModel("openrouter", "~typesafe/jev-latest");
const SESSION_MODEL = getBundledModel("anthropic", "claude-sonnet-4-6");
if (!DECISIONS || !SESSION_MODEL) throw new Error("Expected bundled judge and session models");

const PERMISSION_QUESTION: ChoiceQuestion<"allow" | "ask"> = {
	type: "choice",
	instructions: "Fixture decision only.",
	criteria: { allow: "fixture positive", ask: "fixture negative" },
};

const LEVEL_QUESTION: ChoiceQuestion<"low" | "high"> = {
	type: "choice",
	instructions: "Pick the tier.",
	criteria: { low: "simple", high: "complex" },
};

interface JudgeFixture {
	session: AgentSession;
	requests: Array<{ url: string; body: string }>;
	auth: AuthStorage;
	tempDir: TempDir;
	close(): Promise<void>;
}

/**
 * Build an isolated real session whose judge role resolves to one synthetic
 * native model. Only the fixture registry roster and the global fetch are
 * stubbed; role resolution, credential lookup, native transport, the session
 * journal and the cache fixture are the real implementations.
 */
/** Build an isolated real session whose judge role resolves to one bundled native model. */
async function sessionWithJudge(
	extension: ExtensionFactory,
	options: { judgeRole?: string; models?: readonly Model[] } = {},
): Promise<JudgeFixture> {
	const tempDir = TempDir.createSync("@pi-ext-judgment-");
	const auth = createInMemoryAuthStorage();
	auth.keys.setRuntime("openrouter", "fixture-decisions-key");
	auth.keys.setRuntime("anthropic", "fixture-chat-key");
	const modelRegistry = new ModelRegistry(auth, path.join(tempDir.path(), "models.yml"));
	vi.spyOn(modelRegistry, "getAvailable").mockReturnValue([...(options.models ?? []), SESSION_MODEL, DECISIONS]);
	const requests: Array<{ url: string; body: string }> = [];
	vi.spyOn(globalThis, "fetch").mockImplementation(
		asGlobalFetch(async (url, init) => {
			requests.push({ url: String(url), body: String(init?.body) });
			return Response.json({
				model: "fixture-response-model",
				answers: {
					permission: { type: "choice", choice: "allow", probabilities: { allow: 1, ask: 0 }, confidence: 1 },
					level: { type: "choice", choice: "high", probabilities: { low: 0, high: 1 }, confidence: 1 },
				},
				usage: { input_tokens: 8, output_tokens: 2 },
			});
		}),
	);
	const sessionOptions: Partial<CreateAgentSessionOptions> = {
		cwd: tempDir.path(),
		agentDir: tempDir.path(),
		authStorage: auth,
		modelRegistry,
		model: SESSION_MODEL,
		settings: Settings.isolated({
			modelRoles: { judge: options.judgeRole ?? "openrouter/~typesafe/jev-latest" },
		}),
		sessionManager: SessionManager.inMemory(),
		disableExtensionDiscovery: true,
		extensions: [extension],
		skills: [],
		rules: [],
		contextFiles: [],
		promptTemplates: [],
		slashCommands: [],
		enableMCP: false,
		enableLsp: false,
		skipPythonPreflight: true,
		preloadedCustomToolPaths: [],
		toolNames: ["read"],
	};
	const { session } = await createAgentSession(sessionOptions);
	// `createAgentSession` builds the runner but does not initialize it; the real
	// print/RPC surface does, wiring `ctx.judge` through the session action.
	await initializeExtensions(session, { reportSendError: () => {}, reportRuntimeError: () => {} });
	return {
		session,
		requests,
		auth,
		tempDir,
		close: async () => {
			await session.dispose();
			auth.close();
			tempDir.removeSync();
		},
	};
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe("extension judgment bridge", () => {
	test("an extension uses its session judge and journals only that session", async () => {
		let observed: ExtensionJudgment<Questions> | undefined;
		const fixture = await sessionWithJudge(pi => {
			pi.on("before_agent_start", async (_event, ctx) => {
				if (!ctx.judge) throw new Error("bridge unavailable");
				observed = await ctx.judge(
					{ state: { fixture: "extension-surface" }, questions: { permission: PERMISSION_QUESTION } },
					{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
				);
			});
		});
		try {
			await fixture.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			expect(observed?.route.selected.api).toBe("openrouter-decisions");
			expect(observed?.route.purpose).toContain(":autonomic.permission:");
			const rows = fixture.session.sessionManager.getBranch().filter(entry => entry.type === "model_usage");
			expect(rows.map(entry => entry.purpose)).toEqual([observed!.route.purpose]);
			expect(observed!.route.purpose.endsWith(observed!.route.requestId)).toBe(true);
		} finally {
			await fixture.close();
		}
	});

	test("two same-cwd sessions with different isolated judge roles route differently", async () => {
		const otherJudge = { ...DECISIONS, id: "~typesafe/jev-other", model: undefined } as Model;
		const routed: string[] = [];
		const make = async (model: Model) => {
			const tempDir = TempDir.createSync("@pi-ext-judgment-pair-");
			const auth = createInMemoryAuthStorage();
			auth.keys.setRuntime("openrouter", "fixture-decisions-key");
			auth.keys.setRuntime("anthropic", "fixture-chat-key");
			const registry = new ModelRegistry(auth, path.join(tempDir.path(), "models.yml"));
			vi.spyOn(registry, "getAvailable").mockReturnValue([SESSION_MODEL, model]);
			vi.spyOn(globalThis, "fetch").mockImplementation(
				asGlobalFetch(async () =>
					Response.json({
						model: model.id,
						answers: {
							permission: { type: "choice", choice: "ask", probabilities: { allow: 0, ask: 1 }, confidence: 1 },
						},
						usage: { input_tokens: 1, output_tokens: 1 },
					}),
				),
			);
			const { session } = await createAgentSession({
				cwd: tempDir.path(),
				agentDir: tempDir.path(),
				authStorage: auth,
				modelRegistry: registry,
				model: SESSION_MODEL,
				settings: Settings.isolated({ modelRoles: { judge: `openrouter/${model.id}` } }),
				sessionManager: SessionManager.inMemory(),
				disableExtensionDiscovery: true,
				extensions: [
					pi => {
						pi.on("before_agent_start", async (_event, ctx) => {
							const result = await ctx.judge!(
								{ state: { fixture: "pair" }, questions: { permission: PERMISSION_QUESTION } },
								{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
							);
							routed.push(result.route.selected.model);
						});
					},
				],
				skills: [],
				rules: [],
				contextFiles: [],
				promptTemplates: [],
				slashCommands: [],
				enableMCP: false,
				enableLsp: false,
				skipPythonPreflight: true,
				preloadedCustomToolPaths: [],
				toolNames: ["read"],
			});
			await initializeExtensions(session, { reportSendError: () => {}, reportRuntimeError: () => {} });
			return { session, auth, tempDir };
		};
		const first = await make(DECISIONS);
		const second = await make(otherJudge);
		try {
			// Same cwd is the point: each session's own settings choose its model.
			await first.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			await second.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			expect(routed).toEqual(["~typesafe/jev-latest", "~typesafe/jev-other"]);
		} finally {
			for (const { session, auth, tempDir } of [first, second]) {
				await session.dispose();
				auth.close();
				tempDir.removeSync();
			}
		}
	});

	test("a caller mutating the request after the call still sends the captured bytes", async () => {
		const fixture = await sessionWithJudge(pi => {
			pi.on("before_agent_start", async (_event, ctx) => {
				const request = { state: { fixture: "original-bytes" }, questions: { permission: PERMISSION_QUESTION } };
				const pending = ctx.judge!(request, { purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 });
				request.state.fixture = "mutated-after-capture";
				request.questions.permission.instructions = "mutated instructions";
				await pending;
			});
		});
		try {
			await fixture.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			expect(fixture.requests).toHaveLength(1);
			expect(fixture.requests[0]!.body).toContain("original-bytes");
			expect(fixture.requests[0]!.body).not.toContain("mutated");
		} finally {
			await fixture.close();
		}
	});

	test("the purpose hash is the calling extension's own path, not the caller's purpose", async () => {
		const purposes: string[] = [];
		const probe =
			(label: string): ExtensionFactory =>
			pi => {
				pi.on("before_agent_start", async (_event, ctx) => {
					const result = await ctx.judge!(
						{ state: { fixture: label }, questions: { permission: PERMISSION_QUESTION } },
						{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
					);
					purposes.push(result.route.purpose);
				});
			};
		const fixture = await sessionWithJudge(probe("owner-a"));
		try {
			await fixture.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			// extension:<owner sha256>:<caller purpose>:<request id>
			const [namespace, owner, purpose, requestId] = purposes[0]!.split(":");
			expect(namespace).toBe("extension");
			expect(owner).toMatch(/^[0-9a-f]{64}$/);
			expect(purpose).toBe("autonomic.permission");
			expect(requestId).toMatch(/^[0-9a-f-]{36}$/);
		} finally {
			await fixture.close();
		}
	});

	test("an unknown or non-JSON request rejects before any provider request", async () => {
		let failure: unknown;
		const fixture = await sessionWithJudge(pi => {
			pi.on("before_agent_start", async (_event, ctx) => {
				try {
					await ctx.judge!(
						{ state: { tier: Number.NaN }, questions: { permission: PERMISSION_QUESTION } },
						{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
					);
				} catch (error) {
					failure = error;
				}
			});
		});
		try {
			await fixture.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			expect(failure).toBeInstanceOf(Error);
			expect(fixture.requests).toHaveLength(0);
		} finally {
			await fixture.close();
		}
	});

	test("a session switch settles an in-flight judgment with no successor entry", async () => {
		let settled: "resolved" | "rejected" | undefined;
		const started = Promise.withResolvers<void>();
		const releaseFetch = Promise.withResolvers<void>();
		const fixture = await sessionWithJudge(pi => {
			pi.on("before_agent_start", async (_event, ctx) => {
				const pending = ctx.judge!(
					{ state: { fixture: "switch" }, questions: { permission: PERMISSION_QUESTION } },
					{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 30_000 },
				);
				void pending.then(
					() => {
						settled = "resolved";
					},
					() => {
						settled = "rejected";
					},
				);
				started.resolve();
				await releaseFetch.promise;
			});
		});
		// Hold the provider response so the judgment is genuinely in flight when the session changes.
		vi.spyOn(globalThis, "fetch").mockImplementation(
			asGlobalFetch(async () => {
				await releaseFetch.promise;
				return Response.json({
					model: "fixture-response-model",
					answers: {
						permission: { type: "choice", choice: "allow", probabilities: { allow: 1, ask: 0 }, confidence: 1 },
					},
					usage: { input_tokens: 8, output_tokens: 2 },
				});
			}),
		);
		try {
			const emitting = fixture.session.extensionRunner!.emitBeforeAgentStart("fixture", undefined, []);
			await started.promise;
			// A session change retires the judgment before the handler's own await returns.
			await fixture.session.newSession();
			expect(settled).toBe("rejected");
			// The successor branch carries no entry from the retired judgment.
			const rows = fixture.session.sessionManager.getBranch().filter(entry => entry.type === "model_usage");
			expect(rows).toHaveLength(0);
			releaseFetch.resolve();
			await emitting;
		} finally {
			await fixture.close();
		}
	});

	test("a session without an extension runner exposes no judge", async () => {
		// A hand-built session has no runner (and thus no live invocation scope).
		const fixture = await sessionWithJudge(pi => {
			pi.on("before_agent_start", () => {});
		});
		try {
			const runner = fixture.session.extensionRunner;
			expect(runner).toBeDefined();
			const context = runner!.createContext();
			// Outside a live invocation the accessor is unavailable.
			expect(context.judge).toBeUndefined();
		} finally {
			await fixture.close();
		}
	});
});

describe("captureJson", () => {
	test("copies nested containers, freezing them and never aliasing the source", () => {
		const shared = { tier: "high" };
		const copy = captureJson({ state: { a: shared, b: shared } }) as { state: { a: unknown; b: unknown } };
		expect(copy.state.a).toEqual({ tier: "high" });
		expect(copy.state.b).toEqual({ tier: "high" });
		expect(copy.state.a).not.toBe(shared);
		expect(copy.state.b).not.toBe(shared);
		expect(Object.isFrozen(copy)).toBe(true);
		expect(Object.isFrozen(copy.state)).toBe(true);
		expect(Object.isFrozen(copy.state.a)).toBe(true);
	});

	test("rejects a cycle, a sparse array and an accessor without invoking the getter", () => {
		const cyclic: Record<string, unknown> = { name: "root" };
		cyclic.self = cyclic;
		expect(() => captureJson(cyclic)).toThrow();
		expect(() => captureJson([1, , 3])).toThrow();
		let reads = 0;
		const withAccessor = Object.defineProperty({ tier: "high" }, "danger", {
			enumerable: true,
			get() {
				reads++;
				return "leak";
			},
		});
		expect(() => captureJson(withAccessor)).toThrow();
		expect(reads).toBe(0);
	});

	test("rejects non-JSON values, non-plain prototypes and hidden keys", () => {
		class Holder {
			tier = "high";
		}
		for (const value of [undefined, 1n, Symbol("s"), () => 1, Number.NaN, new Holder(), { a: undefined }]) {
			expect(() => captureJson(value as unknown)).toThrow();
		}
		expect(() => captureJson(Object.defineProperty({}, "hidden", { value: 1, enumerable: false }))).toThrow();
		expect(() => captureJson({ tier: "high", [Symbol("s")]: 1 })).toThrow();
	});
});

describe("settleWithin", () => {
	test("resolves the run result and refuses a missing budget", async () => {
		await expect(settleWithin(async () => 7, 1000, [])).resolves.toBe(7);
		await expect(settleWithin(async () => 1, 0, [])).rejects.toThrow("no budget");
	});

	test("settles on an inherited abort and discards a late completion", async () => {
		const controller = new AbortController();
		controller.abort(new Error("caller cancelled"));
		await expect(settleWithin(async () => 1, 1000, [controller.signal])).rejects.toThrow("caller cancelled");

		let sawAbort = false;
		await expect(
			settleWithin(
				async (signal: AbortSignal) => {
					await new Promise<void>(resolve => {
						signal.addEventListener("abort", () => {
							sawAbort = true;
							resolve();
						});
					});
					return "late";
				},
				20,
				[],
			),
		).rejects.toThrow("judgment cancelled");
		expect(sawAbort).toBe(true);
	});
});

describe("runExtensionJudgment", () => {
	const live = () => ({ signal: new AbortController().signal, current: () => true });
	const deps = () => ({
		settings: Settings.isolated({ modelRoles: { judge: "openrouter/~typesafe/jev-latest" } }),
		registry: (() => {
			const auth = createInMemoryAuthStorage();
			auth.keys.setRuntime("openrouter", "fixture-decisions-key");
			const registry = new ModelRegistry(auth, "/nonexistent/t2-bridge-models.yml");
			vi.spyOn(registry, "getAvailable").mockReturnValue([DECISIONS]);
			return registry;
		})(),
		purpose: "fixture.outer",
	});

	test("bills exactly once and reports the selected route distinctly from the response model", async () => {
		vi.spyOn(globalThis, "fetch").mockImplementation(
			asGlobalFetch(async () =>
				Response.json({
					model: "fixture-response-model",
					answers: { level: { type: "choice", choice: "high", probabilities: { low: 0, high: 1 }, confidence: 1 } },
					usage: { input_tokens: 8, output_tokens: 2 },
				}),
			),
		);
		const usage = vi.fn();
		const result = await runExtensionJudgment(
			{ ...deps(), onUsage: usage },
			{ state: { fixture: "surface" }, questions: { level: LEVEL_QUESTION } },
			{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
			live(),
			"/fixtures/owner-a/extension.ts",
		);
		expect(result.judgment.answers.level.choice).toBe("high");
		expect(result.judgment.model).toBe("fixture-response-model");
		expect(result.route.selected.api).toBe("openrouter-decisions");
		expect(result.route.questions).toBe(1);
		expect(usage).toHaveBeenCalledTimes(1);
	});

	test("derives its owner from the invocation path, never from the caller-supplied purpose", async () => {
		vi.spyOn(globalThis, "fetch").mockImplementation(
			asGlobalFetch(async () =>
				Response.json({
					model: "fixture-response-model",
					answers: { level: { type: "choice", choice: "high", probabilities: { low: 0, high: 1 }, confidence: 1 } },
					usage: { input_tokens: 1, output_tokens: 1 },
				}),
			),
		);
		const options = { purpose: "autonomic.permission", nativeOnly: true as const, timeoutMs: 1000 };
		const request = { state: { fixture: "ownership" }, questions: { level: LEVEL_QUESTION } };
		const a = await runExtensionJudgment(deps(), request, options, live(), "/fixtures/owner-a/extension.ts");
		const b = await runExtensionJudgment(deps(), request, options, live(), "/fixtures/owner-b/extension.ts");
		const ownerOf = (purpose: string) => purpose.split(":")[1];
		expect(ownerOf(a.route.purpose)).not.toBe(ownerOf(b.route.purpose));
		expect(a.route.purpose).not.toContain("/fixtures/owner-b");
	});

	test("rejects a stale live scope and never leaks provider error text", async () => {
		await expect(
			runExtensionJudgment(
				deps(),
				{ state: { fixture: "stale" }, questions: { level: LEVEL_QUESTION } },
				{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
				{ signal: new AbortController().signal, current: () => false },
				"/f/a.ts",
			),
		).rejects.toThrow("unavailable or cancelled");

		const controller = new AbortController();
		vi.spyOn(globalThis, "fetch").mockImplementation(
			asGlobalFetch(
				async (_url, init) =>
					await new Promise<Response>(resolve => {
						init?.signal?.addEventListener("abort", () => resolve(new Response("aborted", { status: 499 })));
					}),
			),
		);
		const pending = runExtensionJudgment(
			deps(),
			{ state: { fixture: "cancel" }, questions: { level: LEVEL_QUESTION } },
			{ purpose: "autonomic.permission", nativeOnly: true, timeoutMs: 1000 },
			{ signal: controller.signal, current: () => true },
			"/f/a.ts",
		);
		controller.abort(new Error("session switched with secret-token"));
		await expect(pending).rejects.toThrow(/^extension judgment [0-9a-f-]+: unavailable or cancelled$/);
	});

	test("keeps the real runner as its dependency (guards a stubbed chain)", () => {
		expect(typeof resolveJudge).toBe("function");
	});
});

describe("extension judgment surface inventory", () => {
	test("a native-only chain rejects when no native candidate is eligible", async () => {
		vi.spyOn(globalThis, "fetch");
		const nonNative = { ...DECISIONS, api: "anthropic-messages", provider: "anthropic" } as Model;
		const auth = createInMemoryAuthStorage();
		auth.keys.setRuntime("anthropic", "fixture-chat-key");
		const registry = new ModelRegistry(auth, "/nonexistent/t2-bridge-models.yml");
		vi.spyOn(registry, "getAvailable").mockReturnValue([nonNative]);
		const routes: JudgmentRoute[] = [];
		await expect(
			resolveJudge({
				settings: Settings.isolated({ modelRoles: { judge: "anthropic/~typesafe/jev-latest" } }),
				registry,
				purpose: "fixture.native-only",
				nativeOnly: true,
				onResult: route => routes.push(route),
			}).judge({ state: { fixture: "reject" }, questions: { level: LEVEL_QUESTION } }),
		).rejects.toThrow();
		expect(routes).toEqual([]);
		auth.close();
	});
});

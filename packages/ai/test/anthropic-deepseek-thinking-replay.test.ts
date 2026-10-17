import { describe, expect, it } from "bun:test";
import { convertAnthropicMessages, streamAnthropic } from "@oh-my-pi/pi-ai/providers/anthropic";
import type {
	AssistantMessage,
	Message,
	Model,
	ModelSpec,
	ToolResultMessage,
	UserMessage,
} from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { classifyModel } from "@oh-my-pi/pi-catalog/compat/taxonomy";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";

/**
 * DeepSeek's thinking mode enforces a wire rule the other anthropic-messages
 * backends do not: every replayed assistant turn must carry reasoning, or the
 * request is refused with HTTP 400 "The reasoning_content in the thinking mode
 * must be passed back to the API". Roughly one short tool-call turn in 4.5
 * returns no reasoning at all, so through a compatibility proxy that maps
 * `thinking` blocks to `reasoning_content` (our CPA lane) such a turn reaches
 * the wire empty and poisons the whole request. Measured on the failing
 * session: 44 of 195 assistant turns had no thinking block; replaying with an
 * empty (`thinking: ""`) block did not help, a one-character block did (10/10).
 *
 * The fix leads every replayed assistant turn of a DeepSeek-class model with a
 * minimal non-empty thinking block when thinking is enabled for the request.
 * Every other model — official Anthropic above all, whose signed thinking and
 * signature replay rules must not move — keeps byte-identical requests.
 */
function makeDeepseekModel(): Model<"anthropic-messages"> {
	return buildModel({
		api: "anthropic-messages",
		provider: "cpa",
		id: "ocgo/deepseek-v4.1-flash",
		name: "DeepSeek V4.1 Flash (CPA · OpenCode Go)",
		baseUrl: "http://100.90.203.21:8317",
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		maxTokens: 384_000,
		contextWindow: 1_000_000,
		reasoning: true,
	} as ModelSpec<"anthropic-messages">);
}

function makeCustomAnthropicModel(): Model<"anthropic-messages"> {
	return buildModel({
		api: "anthropic-messages",
		provider: "custom-anthropic",
		id: "reasoning-model",
		name: "Reasoning Anthropic-Compatible Model",
		baseUrl: "https://llm.example.com/anthropic",
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		maxTokens: 8_192,
		contextWindow: 200_000,
		reasoning: true,
	} as ModelSpec<"anthropic-messages">);
}

/**
 * A DeepSeek-class lane that FORWARDS to signature-enforcing Anthropic:
 * `signingEndpoint: true`, `replayUnsignedThinking: false`. An unsigned
 * `{ signature: "" }` thinking block 400s here ("Invalid signature in
 * thinking"), and a signed empty one must survive untouched.
 */
function makeSigningDeepseekModel(): Model<"anthropic-messages"> {
	return getBundledModel("cloudflare-ai-gateway", "deepseek/deepseek-v4-pro") as Model<"anthropic-messages">;
}

function makeUser(text: string): UserMessage {
	return { role: "user", content: text, timestamp: 0 };
}

function makeAssistant(
	model: Model<"anthropic-messages">,
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "toolUse",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "anthropic-messages",
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: 0,
	};
}

function toolResult(toolCallId: string): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "read",
		content: [{ type: "text", text: "ok" }],
		isError: false,
		timestamp: 0,
	};
}

function toolCallTurn(model: Model<"anthropic-messages">, content?: AssistantMessage["content"]): AssistantMessage {
	return makeAssistant(model, content ?? [{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "x" } }]);
}

interface WireThinkingBlock {
	type: "thinking";
	thinking: string;
	signature?: string;
}
interface WireTextBlock {
	type: "text";
	text: string;
}
interface WireToolUseBlock {
	type: "tool_use";
	id: string;
	name: string;
}
type WireBlock = WireThinkingBlock | WireTextBlock | WireToolUseBlock | { type: string; [key: string]: unknown };

/** Build the real request body through the provider, as a live turn would. */
async function capturePayload(
	model: Model<"anthropic-messages">,
	messages: Message[],
	thinkingEnabled: boolean,
): Promise<{ messages: Array<{ role: string; content: unknown }>; [key: string]: unknown }> {
	const { promise, resolve } = Promise.withResolvers<unknown>();
	streamAnthropic(
		model,
		{ systemPrompt: [], messages },
		{
			apiKey: "sk-ant-api-test",
			signal: AbortSignal.abort(),
			thinkingEnabled,
			// Pin the cache policy: without it `PI_CACHE_RETENTION` in the ambient
			// environment would change the byte fixtures below.
			cacheRetention: "short",
			onPayload: payload => resolve(payload),
		},
	);
	return (await promise) as { messages: Array<{ role: string; content: unknown }> };
}

function assistantBlocks(payload: { messages: Array<{ role: string; content: unknown }> }): WireBlock[] {
	const assistant = payload.messages.find(message => message.role === "assistant");
	expect(assistant).toBeDefined();
	return assistant?.content as WireBlock[];
}

/** A thinking-less tool-call turn sandwiched between user turns. */
function unthoughtToolCallHistory(model: Model<"anthropic-messages">): Message[] {
	return [makeUser("go"), toolCallTurn(model), toolResult("call_1"), makeUser("next")];
}

/**
 * Whole-request pins captured before the fix on the same build. The change is
 * gated on the DeepSeek model class, so a Claude or custom anthropic-messages
 * request has to serialize exactly as it did before. The literal couples to
 * bundled catalog data (`max_tokens`, the model's thinking mode), so a
 * `models.json` change for these two models is expected to move it.
 */
const CLAUDE_REQUEST_FIXTURE = `{"model":"claude-opus-4-6","messages":[{"role":"user","content":"go"},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read","input":{"path":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"ok"}],"is_error":false,"cache_control":{"type":"ephemeral"}}]},{"role":"user","content":[{"type":"text","text":"next","cache_control":{"type":"ephemeral"}}]}],"max_tokens":128000,"thinking":{"type":"adaptive"},"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},"stream":true}`;
const CUSTOM_REQUEST_FIXTURE = `{"model":"reasoning-model","messages":[{"role":"user","content":"go"},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read","input":{"path":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok","is_error":false,"cache_control":{"type":"ephemeral"}}]},{"role":"user","content":[{"type":"text","text":"next","cache_control":{"type":"ephemeral"}}]}],"max_tokens":8192,"thinking":{"type":"enabled","budget_tokens":1024,"display":"summarized"},"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},"stream":true}`;

describe("DeepSeek thinking-mode replay over anthropic-messages", () => {
	it("leads a reasoning-less tool-call turn with a non-empty thinking block", async () => {
		const model = makeDeepseekModel();
		const payload = await capturePayload(model, unthoughtToolCallHistory(model), true);
		const blocks = assistantBlocks(payload);
		expect(blocks[0]?.type).toBe("thinking");
		expect((blocks[0] as WireThinkingBlock).thinking.length).toBeGreaterThan(0);
		// The tool_use that the turn exists for must survive behind the placeholder.
		expect(blocks.filter(block => block.type === "tool_use")).toHaveLength(1);
		expect(blocks.findIndex(block => block.type === "tool_use")).toBe(1);
	});

	it("leads a reasoning-less text-only turn with a non-empty thinking block", async () => {
		const model = makeDeepseekModel();
		const messages: Message[] = [
			makeUser("go"),
			makeAssistant(model, [{ type: "text", text: "Done." }], "stop"),
			makeUser("again"),
		];
		const blocks = assistantBlocks(await capturePayload(model, messages, true));
		expect(blocks[0]?.type).toBe("thinking");
		expect((blocks[0] as WireThinkingBlock).thinking.length).toBeGreaterThan(0);
		expect(blocks.some(block => block.type === "text" && (block as WireTextBlock).text === "Done.")).toBe(true);
	});

	it("treats an empty-text thinking block as no reasoning and leaves no empty block behind", async () => {
		const model = makeDeepseekModel();
		const payload = await capturePayload(
			model,
			[
				makeUser("go"),
				toolCallTurn(model, [
					{ type: "thinking", thinking: "", thinkingSignature: "" },
					{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "x" } },
				]),
				toolResult("call_1"),
				makeUser("next"),
			],
			true,
		);
		const blocks = assistantBlocks(payload);
		const thinkingBlocks = blocks.filter(block => block.type === "thinking") as WireThinkingBlock[];
		expect(thinkingBlocks).toHaveLength(1);
		expect(thinkingBlocks[0]?.thinking.length).toBeGreaterThan(0);
		expect(blocks[0]?.type).toBe("thinking");
	});

	it("leaves a turn that already carries reasoning unchanged", async () => {
		const model = makeDeepseekModel();
		const payload = await capturePayload(
			model,
			[
				makeUser("go"),
				toolCallTurn(model, [
					{ type: "thinking", thinking: "plan: read the file", thinkingSignature: "" },
					{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "x" } },
				]),
				toolResult("call_1"),
				makeUser("next"),
			],
			true,
		);
		const blocks = assistantBlocks(payload);
		const thinkingBlocks = blocks.filter(block => block.type === "thinking") as WireThinkingBlock[];
		expect(thinkingBlocks).toHaveLength(1);
		expect(thinkingBlocks[0]?.thinking).toBe("plan: read the file");
		expect(blocks[0]?.type).toBe("thinking");
	});

	it("adds nothing when thinking is disabled for the request", async () => {
		const model = makeDeepseekModel();
		const blocks = assistantBlocks(await capturePayload(model, unthoughtToolCallHistory(model), false));
		expect(blocks.some(block => block.type === "thinking")).toBe(false);
		expect(blocks[0]?.type).toBe("tool_use");
	});

	it("leaves a signing DeepSeek lane without a placeholder", async () => {
		// DeepSeek by class, but the lane forwards to signature-enforcing
		// Anthropic (replayUnsignedThinking: false). An unsigned `{ signature: "" }`
		// block 400s there ("Invalid signature in thinking"), so nothing may be added.
		const model = makeSigningDeepseekModel();
		const blocks = assistantBlocks(await capturePayload(model, unthoughtToolCallHistory(model), true));
		expect(blocks.some(block => block.type === "thinking")).toBe(false);
		expect(blocks[0]?.type).toBe("tool_use");
	});

	it("keeps a signed empty thinking block untouched on a signing DeepSeek lane", async () => {
		const model = makeSigningDeepseekModel();
		const payload = await capturePayload(
			model,
			[
				makeUser("go"),
				toolCallTurn(model, [
					{ type: "thinking", thinking: "", thinkingSignature: "sig_cloudflare" },
					{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "x" } },
				]),
				toolResult("call_1"),
				makeUser("next"),
			],
			true,
		);
		const blocks = assistantBlocks(payload);
		const thinkingBlocks = blocks.filter(block => block.type === "thinking") as WireThinkingBlock[];
		expect(thinkingBlocks).toHaveLength(1);
		expect(thinkingBlocks[0]?.thinking).toBe("");
		expect(thinkingBlocks[0]?.signature).toBe("sig_cloudflare");
	});

	it("adds nothing when the caller drops all thinking", () => {
		// The session latch that learns from a signing 400 sets dropAllThinking;
		// it must not be re-populated with the DeepSeek placeholder.
		const model = makeDeepseekModel();
		const params = convertAnthropicMessages(unthoughtToolCallHistory(model), model, false, {
			dropAllThinking: true,
			ensureAssistantThinking: true,
		});
		const assistant = params.find(param => param.role === "assistant");
		const blocks = (assistant?.content ?? []) as WireBlock[];
		expect(blocks.some(block => block.type === "thinking")).toBe(false);
		expect(blocks[0]?.type).toBe("tool_use");
	});

	it("classifies the production lane as DeepSeek", () => {
		expect(classifyModel("cpa", "ocgo/deepseek-v4.1-flash").class).toBe("deepseek");
	});

	it("leads every reasoning-less turn of a mixed history", async () => {
		const model = makeDeepseekModel();
		const messages: Message[] = [
			makeUser("go"),
			makeAssistant(
				model,
				[
					{ type: "thinking", thinking: "first hop plan", thinkingSignature: "" },
					{ type: "text", text: "First answer." },
				],
				"stop",
			),
			makeUser("again"),
			toolCallTurn(model),
			toolResult("call_1"),
			makeUser("next"),
		];
		const payload = await capturePayload(model, messages, true);
		const assistantParams = payload.messages.filter(message => message.role === "assistant");
		expect(assistantParams).toHaveLength(2);
		const [thoughtTurn, unthoughtTurn] = assistantParams.map(param => param.content as WireBlock[]);
		const thoughtBlocks = thoughtTurn ?? [];
		const unthoughtBlocks = unthoughtTurn ?? [];
		// The first turn already had reasoning: its text is untouched, no placeholder.
		expect((thoughtBlocks[0] as WireThinkingBlock).thinking).toBe("first hop plan");
		expect(thoughtBlocks.filter(block => block.type === "thinking")).toHaveLength(1);
		// The second turn had none: it now leads with the placeholder before its tool_use.
		expect(unthoughtBlocks[0]?.type).toBe("thinking");
		expect((unthoughtBlocks[0] as WireThinkingBlock).thinking.length).toBeGreaterThan(0);
		expect(unthoughtBlocks.findIndex(block => block.type === "tool_use")).toBe(1);
	});

	it("keeps a Claude request byte-identical", async () => {
		const model = getBundledModel("anthropic", "claude-opus-4-6") as Model<"anthropic-messages">;
		const payload = await capturePayload(model, unthoughtToolCallHistory(model), true);
		expect(JSON.stringify(payload)).toBe(CLAUDE_REQUEST_FIXTURE);
	});

	it("keeps a non-DeepSeek custom anthropic-messages request byte-identical", async () => {
		const model = makeCustomAnthropicModel();
		const payload = await capturePayload(model, unthoughtToolCallHistory(model), true);
		expect(JSON.stringify(payload)).toBe(CUSTOM_REQUEST_FIXTURE);
	});
});

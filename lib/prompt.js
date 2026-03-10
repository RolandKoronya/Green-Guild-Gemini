// lib/prompt.js
import fs from "fs";
import path from "path";

const PROMPT_FILE = process.env.PROMPT_FILE || "base.en.md";
const PROMPT_PATH = path.join(process.cwd(), "prompts", PROMPT_FILE);

// Simple in-memory cache
let CACHE = { text: "", mtimeMs: 0 };

function readPromptFile() {
  try {
    const stat = fs.statSync(PROMPT_PATH);

    if (!CACHE.text || stat.mtimeMs !== CACHE.mtimeMs) {
      CACHE.text = fs.readFileSync(PROMPT_PATH, "utf8");
      CACHE.mtimeMs = stat.mtimeMs;
      console.log(
        `[prompts] Loaded ${PROMPT_FILE} (${Math.round(CACHE.text.length / 1024)} KB)`
      );
    }

    return CACHE.text;
  } catch (e) {
    console.warn(
      `[prompts] Could not read prompts/${PROMPT_FILE}, using fallback.`
    );
    return "You are The Green Tutor — friendly, thoughtful, precise, and practical. Reply in English.";
  }
}

export function buildSystemPrompt() {
  return readPromptFile();
}

export function invalidatePromptCache() {
  CACHE = { text: "", mtimeMs: 0 };
  console.log("[prompts] Cache invalidated.");
}
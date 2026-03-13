// server.js - THE GREEN TUTOR (English Edition)

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Firestore } from "@google-cloud/firestore";
import fs from "fs";
import path from "path";

import { loadKB } from "./lib/kb_loader.js";
import { createRetriever } from "./lib/retriever.js";

dotenv.config();

const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "10mb" }));

// -----------------------------
// Firestore (local + Cloud Run safe)
// -----------------------------
const firestoreConfig = {
  projectId: "green-tutor",
  databaseId: "(default)",
};

if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
  firestoreConfig.keyFilename = process.env.GOOGLE_APPLICATION_CREDENTIALS;
}

const db = new Firestore(firestoreConfig);

// -----------------------------
// CORS
// -----------------------------
const allowedOrigins = [
  "http://localhost:5173",
  "http://localhost:3000",
  "https://greenguild.co.uk",
  "https://www.greenguild.co.uk",
];

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      return callback(new Error("Not allowed by CORS"));
    },
  })
);

// -----------------------------
// Rate limit
// -----------------------------
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
});
app.use(limiter);

// -----------------------------
// Auth
// -----------------------------
const PUBLIC_API_TOKEN =
  process.env.PUBLIC_API_TOKEN || "greentutor-secure-token";

function auth(req, res, next) {
  const token = req.headers["x-client-token"] || "";
  if (token !== PUBLIC_API_TOKEN) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  return next();
}

// -----------------------------
// AI clients
// -----------------------------
if (!process.env.GEMINI_API_KEY) {
  console.warn("⚠️ GEMINI_API_KEY is missing.");
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const CHAT_MODEL_NAME = "gemini-2.5-flash";
const SEARCH_HELPER_MODEL = "gemini-2.5-flash";
const CLARIFIER_MODEL_NAME = "gemini-2.5-flash";

// -----------------------------
// Prompt loader
// -----------------------------
const PROMPT_FILE = process.env.PROMPT_FILE || "base.en.md";
const PROMPT_PATH = path.join(process.cwd(), "prompts", PROMPT_FILE);
let cachedSystemPrompt = null;

function buildSystemPrompt() {
  if (!cachedSystemPrompt) {
    try {
      cachedSystemPrompt = fs.readFileSync(PROMPT_PATH, "utf8");
      console.log(`✅ Prompt loaded: ${PROMPT_FILE}`);
    } catch (e) {
      console.warn(
        `[prompt] Could not read ${PROMPT_FILE}, using fallback prompt.`
      );
      cachedSystemPrompt =
        "You are The Green Tutor — a thoughtful, practical herbal mentor. Reply in English using British spelling.";
    }
  }
  return cachedSystemPrompt;
}

// -----------------------------
// Memory / Session
// -----------------------------
const MAX_CONTEXT = 24;
const MAX_STORAGE = 100;

function getConversationKey(req) {
  const userId = req.headers["x-user-id"];
  const sessionId = req.headers["x-session-id"];
  const id = userId
    ? `user:${userId}`
    : sessionId
    ? `session:${sessionId}`
    : `ip:${req.ip}`;

  return id.replace(/\//g, "_");
}

async function loadSession(key) {
  const doc = await db.collection("sessions_en").doc(key).get();
  return doc.exists ? doc.data().messages || [] : [];
}

async function saveSession(key, messages) {
  await db.collection("sessions_en").doc(key).set({
    messages,
    updatedAt: new Date(),
  });
}

// -----------------------------
// Structured user profile
// -----------------------------
function emptyProfile() {
  return {
    constitution: "",
    allergies: "",
    medications: "",
    preferredPreparations: "",
    goals: "",
    notes: "",
  };
}

async function loadUserProfile(key) {
  const doc = await db.collection("profiles_en").doc(key).get();

  if (!doc.exists) return emptyProfile();

  const data = doc.data() || {};

  return {
    constitution: data.constitution || "",
    allergies: data.allergies || "",
    medications: data.medications || "",
    preferredPreparations: data.preferredPreparations || "",
    goals: data.goals || "",
    notes: data.notes || "",
  };
}

async function saveUserProfile(key, profile) {
  const clean = {
    constitution: profile?.constitution || "",
    allergies: profile?.allergies || "",
    medications: profile?.medications || "",
    preferredPreparations: profile?.preferredPreparations || "",
    goals: profile?.goals || "",
    notes: profile?.notes || "",
    updatedAt: new Date(),
  };

  await db.collection("profiles_en").doc(key).set(clean, { merge: true });
}

function buildProfileBlock(profile) {
  if (!profile) return "";

  const lines = [
    profile.constitution ? `- Constitution: ${profile.constitution}` : "",
    profile.allergies ? `- Allergies: ${profile.allergies}` : "",
    profile.medications ? `- Medications: ${profile.medications}` : "",
    profile.preferredPreparations
      ? `- Preferred preparations: ${profile.preferredPreparations}`
      : "",
    profile.goals ? `- Goals: ${profile.goals}` : "",
    profile.notes ? `- Notes: ${profile.notes}` : "",
  ].filter(Boolean);

  if (!lines.length) return "";
  return `USER PROFILE:\n${lines.join("\n")}`;
}

// -----------------------------
// Knowledge base + retriever
// -----------------------------
let kb = null;
let retriever = null;
let kbLoadError = null;

try {
  kb = loadKB(path.join(process.cwd(), "kb_en"));
  retriever = createRetriever(kb, {
    geminiApiKey: process.env.GEMINI_API_KEY,
  });
  console.log("✅ KB loaded successfully");
} catch (e) {
  kbLoadError = e;
  console.error("❌ KB load failed:", e);
}

// -----------------------------
// Query expansion
// -----------------------------
async function expandQueryWithAI(userQuery) {
  const fastModel = genAI.getGenerativeModel({ model: SEARCH_HELPER_MODEL });

  const result = await fastModel.generateContent(
    `Identify herbal or botanical terms in: "${userQuery}". Provide Latin scientific names where relevant. Return ONLY keywords.`
  );

  return `${userQuery} ${result.response.text()}`;
}

// -----------------------------
// Clarification + risk tier
// -----------------------------
function safeJsonParse(text, fallback = null) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function recentHistoryForClassifier(dbHistory) {
  return dbHistory
    .slice(-6)
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");
}

function normaliseRiskTier(tier) {
  if (tier === "green" || tier === "yellow" || tier === "red") return tier;
  return "green";
}

async function getGuidanceDecision({
  userText,
  userProfile,
  dbHistory,
}) {
  if (!userText || !userText.trim()) {
    return {
      shouldAsk: false,
      question: "",
      reason: "none",
      riskTier: "green",
    };
  }

  const model = genAI.getGenerativeModel({ model: CLARIFIER_MODEL_NAME });

  const profileBlock = buildProfileBlock(userProfile) || "USER PROFILE:\n- none";
  const historyBlock =
    recentHistoryForClassifier(dbHistory) || "No recent history.";

  const prompt = `
You are deciding how a herbal mentor should respond.

Return ONLY valid JSON in exactly this format:
{
  "shouldAsk": true,
  "question": "Your question here",
  "reason": "safety|goal|context|form|person|none",
  "riskTier": "green|yellow|red"
}

Definitions:
- green = low-risk educational question, answer normally
- yellow = personalised but caution-worthy, answer usefully with examples if appropriate, then brief caution at the end
- red = acute/high-risk/medically sensitive, do not give a direct personal prescription, formula, tea recipe, or regimen, but still give useful educational reasoning and brief caution at the end

Rules:
- Ask a clarifying question only if the missing information would materially improve safety, accuracy, or relevance.
- Ask at most ONE question.
- Do not ask broad filler questions like "Can you tell me more?"
- Do not ask for information already known from the user profile or recent history.
- Prefer direct answers for general educational questions.
- If the user is asking for direct personal use in a red-tier scenario, usually do NOT ask more questions unless one missing fact is essential. Instead, allow a red-tier educational response.
- If no clarifying question is needed, return:
  {"shouldAsk": false, "question": "", "reason": "none", "riskTier": "green"}

Examples of red-tier situations:
- high fever with significant symptoms
- warfarin or anticoagulants plus direct personal recipe/request
- multiple prescription medicines plus direct formula request
- urgent-seeming serious symptoms
- pregnancy/breastfeeding/children with significant symptoms and direct personal remedy request

Examples of yellow-tier situations:
- asking what herbs might suit their profile
- asking which herbs fit their constitution
- asking personal-use questions where caution matters but a useful educational answer is still clearly possible

${profileBlock}

RECENT HISTORY:
${historyBlock}

USER MESSAGE:
${userText}
`;

  try {
    const result = await model.generateContent(prompt);
    const raw = result.response.text().trim();

    const parsed =
      safeJsonParse(raw) ||
      safeJsonParse(raw.replace(/```json|```/g, "").trim());

    if (
      parsed &&
      typeof parsed.shouldAsk === "boolean" &&
      typeof parsed.question === "string" &&
      typeof parsed.reason === "string"
    ) {
      return {
        shouldAsk: parsed.shouldAsk,
        question: parsed.question || "",
        reason: parsed.reason || "none",
        riskTier: normaliseRiskTier(parsed.riskTier),
      };
    }

    return {
      shouldAsk: false,
      question: "",
      reason: "none",
      riskTier: "green",
    };
  } catch (e) {
    console.error("Guidance decision failed:", e);
    return {
      shouldAsk: false,
      question: "",
      reason: "none",
      riskTier: "green",
    };
  }
}

function buildRiskInstruction(riskTier) {
  if (riskTier === "yellow") {
    return `
RISK TIER: YELLOW

This is a personalised but caution-worthy case.
Give a useful educational answer.
Use the user's profile actively.
Give candidate herbs or examples where appropriate if supported by the provided material.
Explain why they fit the pattern.
Avoid overconfident safety claims.
Avoid precise dosage unless clearly safe and grounded in the provided material.
Place any caution briefly at the end.
Do not let the caution swallow the whole answer.
`;
  }

  if (riskTier === "red") {
    return `
RISK TIER: RED

This is an acute, high-risk, or medically sensitive case.
Do NOT give a direct personal prescription, formula, tea recipe, or regimen.
Do NOT say "here is what would work for you personally."
However, do NOT collapse into a useless refusal.
Still provide useful educational reasoning:
- explain why the case is more cautious
- explain how an herbalist would think about the pattern
- discuss broad categories, formulation logic, gentle conceptual options, or relevant examples from the provided material when appropriate
- avoid direct personal recipe language
- end with a short, practical safety note
Do not make "talk to your GP" the whole answer.
`;
  }

  return `
RISK TIER: GREEN

This is a low-risk educational case.
Answer normally and directly.
`;
}

// -----------------------------
// Endpoints
// -----------------------------
app.get("/health", (req, res) => {
  res.json({
    ok: true,
    service: "The Green Tutor API",
    kbLoaded: !!retriever,
    kbError: kbLoadError ? String(kbLoadError.message || kbLoadError) : null,
  });
});

app.get("/get-profile", auth, async (req, res) => {
  try {
    const profile = await loadUserProfile(getConversationKey(req));
    res.json({ ok: true, profile });
  } catch (e) {
    console.error("Profile load error:", e);
    res.status(500).json({ error: "Failed to load profile." });
  }
});

app.post("/update-profile", auth, async (req, res) => {
  try {
    await saveUserProfile(getConversationKey(req), req.body.profile || {});
    res.json({ ok: true });
  } catch (e) {
    console.error("Profile update error:", e);
    res.status(500).json({ error: "Failed to update profile." });
  }
});

app.get("/history", auth, async (req, res) => {
  try {
    const hist = await loadSession(getConversationKey(req));
    const messages = hist.map((m) => ({
      who: m.role === "user" ? "user" : "bot",
      text: m.content,
    }));
    res.json({ ok: true, messages });
  } catch (e) {
    console.error("History load error:", e);
    res.status(500).json({ error: "Failed to load history." });
  }
});

app.post("/chat", auth, async (req, res) => {
  try {
    const {
      message: userText,
      image: imageBase64,
      mimeType: imageMime,
    } = req.body;

    if (!userText && !imageBase64) {
      return res.status(400).json({ error: "Message or image is required." });
    }

    const convKey = getConversationKey(req);
    const dbHistory = await loadSession(convKey);
    const userProfile = await loadUserProfile(convKey);

    let decision = {
      shouldAsk: false,
      question: "",
      reason: "none",
      riskTier: "green",
    };

    if (userText && !imageBase64) {
      decision = await getGuidanceDecision({
        userText,
        userProfile,
        dbHistory,
      });

      if (decision.shouldAsk && decision.question.trim()) {
        await saveSession(
          convKey,
          [
            ...dbHistory,
            { role: "user", content: userText },
            { role: "assistant", content: decision.question.trim() },
          ].slice(-MAX_STORAGE)
        );

        return res.json({
          ok: true,
          answer: decision.question.trim(),
          meta: {
            askedClarifyingQuestion: true,
            reason: decision.reason,
            riskTier: decision.riskTier,
          },
        });
      }
    }

    const expandedQuery = userText
      ? await expandQueryWithAI(userText)
      : "plant image herbal identification";

    const hits = retriever
      ? await retriever.search(expandedQuery, { k: 6 })
      : [];

    const contextBlock =
      hits.length > 0
        ? `\n\nKNOWLEDGE BASE DATA:\n${hits.map((h) => h.text).join("\n")}`
        : "";

    const profileBlock = buildProfileBlock(userProfile);
    const formattedProfileBlock = profileBlock ? `\n\n${profileBlock}` : "";

    const kbFallbackBlock =
      !retriever && kbLoadError
        ? `\n\nNOTE: The knowledge base is currently unavailable, so answer as carefully as possible without citing unavailable internal material.`
        : "";

    const riskInstruction = buildRiskInstruction(decision.riskTier);

    const finalInstruction = `${buildSystemPrompt()}

${riskInstruction}${formattedProfileBlock}${contextBlock}${kbFallbackBlock}`;

    let recentHistory = dbHistory.slice(-MAX_CONTEXT).map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content || "" }],
    }));

    while (recentHistory.length > 0 && recentHistory[0].role !== "user") {
      recentHistory.shift();
    }

    const model = genAI.getGenerativeModel({
      model: CHAT_MODEL_NAME,
      systemInstruction: finalInstruction,
      safetySettings: [
        {
          category: "HARM_CATEGORY_DANGEROUS_CONTENT",
          threshold: "BLOCK_NONE",
        },
      ],
    });

    const chat = model.startChat({ history: recentHistory });

    const result = imageBase64
      ? await chat.sendMessage([
          { text: userText || "Analyse this image:" },
          {
            inlineData: {
              mimeType: imageMime,
              data: imageBase64,
            },
          },
        ])
      : await chat.sendMessage(userText);

    const reply = result.response.text();

    await saveSession(
      convKey,
      [
        ...dbHistory,
        { role: "user", content: userText || "[Image uploaded]" },
        { role: "assistant", content: reply },
      ].slice(-MAX_STORAGE)
    );

    res.json({
      ok: true,
      answer: reply,
      meta: {
        askedClarifyingQuestion: false,
        riskTier: decision.riskTier,
      },
    });
  } catch (e) {
    console.error("Chat error:", e);

    let errorMsg = "An error occurred while generating a response.";

    if (e.message?.includes("PROHIBITED_CONTENT")) {
      errorMsg =
        "Response blocked by safety filters. Please rephrase your request.";
    } else if (e.status === 413) {
      errorMsg = "Image too large (max 10MB).";
    }

    res.status(500).json({ error: errorMsg });
  }
});

// -----------------------------
// Start server
// -----------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ The Green Tutor active on port ${PORT}`);
});
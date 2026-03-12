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
// Clarification gate
// -----------------------------
function safeJsonParse(text, fallback = null) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function recentHistoryForClarifier(dbHistory) {
  return dbHistory
    .slice(-6)
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");
}

async function getClarificationDecision({
  userText,
  userProfile,
  dbHistory,
}) {
  if (!userText || !userText.trim()) {
    return { shouldAsk: false, question: "", reason: "none" };
  }

  const model = genAI.getGenerativeModel({ model: CLARIFIER_MODEL_NAME });

  const profileBlock = buildProfileBlock(userProfile) || "USER PROFILE:\n- none";
  const historyBlock =
    recentHistoryForClarifier(dbHistory) || "No recent history.";

  const prompt = `
You are deciding whether a herbal mentor should ask ONE clarifying question before answering.

Return ONLY valid JSON in exactly this format:
{
  "shouldAsk": true,
  "question": "Your question here",
  "reason": "safety|goal|context|form|person|none"
}

Rules:
- Ask a question only if the missing information would materially improve safety, accuracy, or relevance.
- Ask at most ONE question.
- Do not ask broad filler questions like "Can you tell me more?"
- Do not ask for something already known from the user profile or recent history.
- Prefer answering directly for general educational questions.
- For dosage, herb-drug interactions, pregnancy, breastfeeding, children, prescription medicines, or serious illness, prioritise safety.
- If no clarifying question is needed, return:
  {"shouldAsk": false, "question": "", "reason": "none"}

Examples of good clarifying questions:
- "Is this for you, or for someone else?"
- "Do you mean tea, tincture, or capsules?"
- "Are any prescription medicines involved here?"
- "Are you asking about the herb generally, or about using it in practice?"
- "Is the main issue more dryness, stagnation, tension, or irritation?"

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
      return parsed;
    }

    return { shouldAsk: false, question: "", reason: "none" };
  } catch (e) {
    console.error("Clarification decision failed:", e);
    return { shouldAsk: false, question: "", reason: "none" };
  }
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

    // Ask a clarifying question only for text-based messages
    if (userText && !imageBase64) {
      const clarification = await getClarificationDecision({
        userText,
        userProfile,
        dbHistory,
      });

      if (clarification.shouldAsk && clarification.question.trim()) {
        await saveSession(
          convKey,
          [
            ...dbHistory,
            { role: "user", content: userText },
            { role: "assistant", content: clarification.question.trim() },
          ].slice(-MAX_STORAGE)
        );

        return res.json({
          ok: true,
          answer: clarification.question.trim(),
          meta: {
            askedClarifyingQuestion: true,
            reason: clarification.reason,
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

    const finalInstruction = `${buildSystemPrompt()}${formattedProfileBlock}${contextBlock}${kbFallbackBlock}`;

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
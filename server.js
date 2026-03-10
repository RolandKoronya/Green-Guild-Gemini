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

// Safer production model choice for Cloud Run deployment
const CHAT_MODEL_NAME = "gemini-2.5-flash";
const SEARCH_HELPER_MODEL = "gemini-2.5-flash";

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
// Memory
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

async function loadUserProfile(key) {
  const doc = await db.collection("profiles_en").doc(key).get();
  return doc.exists ? doc.data().bio || "" : "";
}

async function saveUserProfile(key, bioText) {
  await db
    .collection("profiles_en")
    .doc(key)
    .set(
      {
        bio: bioText,
        updatedAt: new Date(),
      },
      { merge: true }
    );
}

// -----------------------------
// Knowledge base + retriever
// -----------------------------
const kb = loadKB(path.join(process.cwd(), "kb_en"));
const retriever = createRetriever(kb, {
  geminiApiKey: process.env.GEMINI_API_KEY,
});

async function expandQueryWithAI(userQuery) {
  const fastModel = genAI.getGenerativeModel({ model: SEARCH_HELPER_MODEL });

  const result = await fastModel.generateContent(
    `Identify herbal or botanical terms in: "${userQuery}". Provide Latin scientific names where relevant. Return ONLY keywords.`
  );

  return `${userQuery} ${result.response.text()}`;
}

// -----------------------------
// Endpoints
// -----------------------------
app.get("/health", (req, res) => {
  res.json({ ok: true, service: "The Green Tutor API" });
});

app.get("/get-profile", auth, async (req, res) => {
  try {
    const bio = await loadUserProfile(getConversationKey(req));
    res.json({ ok: true, bio });
  } catch (e) {
    console.error("Profile load error:", e);
    res.status(500).json({ error: "Failed to load profile." });
  }
});

app.post("/update-profile", auth, async (req, res) => {
  try {
    await saveUserProfile(getConversationKey(req), req.body.bio || "");
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
    const userBio = await loadUserProfile(convKey);

    const expandedQuery = userText
      ? await expandQueryWithAI(userText)
      : "plant image herbal identification";

    const hits = await retriever.search(expandedQuery, { k: 6 });

    const contextBlock =
      hits.length > 0
        ? `\n\nKNOWLEDGE BASE DATA:\n${hits.map((h) => h.text).join("\n")}`
        : "";

    const bioBlock = userBio
      ? `\n\nUSER PROFILE (Remember these facts):\n${userBio}`
      : "";

    const finalInstruction = `${buildSystemPrompt()}${bioBlock}${contextBlock}`;

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

    if (userText && userText.toLowerCase().includes("remember this")) {
      const updatedBio =
        (userBio ? userBio + "\n" : "") +
        "- " +
        userText.replace(/remember this/gi, "").trim();
      await saveUserProfile(convKey, updatedBio);
    }

    res.json({ ok: true, answer: reply });
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
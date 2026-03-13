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

function normaliseText(value) {
  return String(value || "").toLowerCase();
}

function combinedProfileText(profile) {
  if (!profile) return "";
  return [
    profile.constitution,
    profile.allergies,
    profile.medications,
    profile.preferredPreparations,
    profile.goals,
    profile.notes,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function countMedicationItems(medicationsText) {
  const raw = String(medicationsText || "").trim();
  if (!raw) return 0;

  return raw
    .split(/,|;|\n|\/|\band\b|\&/gi)
    .map((s) => s.trim())
    .filter(Boolean).length;
}

function detectSafetyFlags(profile = {}) {
  const meds = normaliseText(profile.medications);
  const allergies = normaliseText(profile.allergies);
  const notes = normaliseText(profile.notes);
  const constitution = normaliseText(profile.constitution);
  const goals = normaliseText(profile.goals);

  const haystack = [meds, allergies, notes, constitution, goals]
    .filter(Boolean)
    .join(" ");

  const flags = [];

  if (
    /\bwarfarin\b|\banticoagulant\b|\bblood thinner\b|\bapixaban\b|\brivaroxaban\b|\bedoxaban\b|\bdabigatran\b|\bclopidogrel\b|\bantiplatelet\b/.test(
      haystack
    )
  ) {
    flags.push("anticoagulant_or_antiplatelet");
  }

  if (/\bpregnan/.test(haystack)) {
    flags.push("pregnancy");
  }

  if (/\bbreast[\s-]?feeding\b|\bnursing\b|\blactation\b/.test(haystack)) {
    flags.push("breastfeeding");
  }

  if (/\bseizure\b|\bepilep/.test(haystack)) {
    flags.push("seizure_risk");
  }

  if (/\btransplant\b|\btacrolimus\b|\bcyclosporin\b|\bcyclosporine\b/.test(haystack)) {
    flags.push("transplant_medicines");
  }

  if (/\bliver disease\b|\bcirrhosis\b|\bhepatitis\b|\bimpaired liver\b/.test(haystack)) {
    flags.push("serious_liver_disease");
  }

  if (/\bkidney disease\b|\brenal\b|\bckd\b|\bimpaired kidney\b/.test(haystack)) {
    flags.push("serious_kidney_disease");
  }

  if (
    /\banaphyl/.test(allergies) ||
    /\bsevere allerg/.test(allergies) ||
    /\bepi[\s-]?pen\b/.test(allergies)
  ) {
    flags.push("severe_allergy");
  }

  if (countMedicationItems(profile.medications) >= 2) {
    flags.push("multiple_prescription_medicines");
  }

  return Array.from(new Set(flags));
}

function isRemedyStyleQuestion(userText) {
  const q = normaliseText(userText);

  return (
    /\btea\b|\btisane\b|\binfusion\b|\bdecoction\b|\btincture\b|\bblend\b|\bformula\b|\brecipe\b|\bremedy\b|\bherb(s)?\b|\bplant(s)?\b|\bwhat should i take\b|\bwhat can i take\b|\bwhat would you use\b|\bwhat do you recommend\b|\bgive me\b.*\btea\b|\bgive me\b.*\brecipe\b|\bmake me\b.*\bblend\b/.test(
      q
    ) ||
    /\bfor anxiety\b|\bfor sleep\b|\bfor stress\b|\bfor digestion\b|\bfor pain\b|\bfor cold\b|\bfor flu\b/.test(
      q
    )
  );
}

function isDirectPersonalUseRequest(userText) {
  const q = normaliseText(userText);

  return /\bi have\b|\bmy\b|\bfor me\b|\bbased on my profile\b|\bgiven my\b|\bcan i take\b|\bwhat should i take\b|\bgive me\b|\brecipe\b|\bblend\b/.test(
    q
  );
}

function buildSafetyOverride({ userText, userProfile, currentRiskTier }) {
  const flags = detectSafetyFlags(userProfile);
  const remedyStyle = isRemedyStyleQuestion(userText);
  const personalUse = isDirectPersonalUseRequest(userText);

  const hasMajorSafetyFactor = flags.length > 0;
  const shouldForceRed = hasMajorSafetyFactor && remedyStyle && personalUse;

  let riskTier = currentRiskTier || "green";
  if (shouldForceRed) riskTier = "red";

  let extraInstruction = "";
  if (shouldForceRed) {
    extraInstruction = `
SAFETY OVERRIDE: HIGH-RISK PROFILE + REMEDY REQUEST

The user profile contains one or more major safety factors:
${flags.map((f) => `- ${f}`).join("\n")}

Because the user is asking for a remedy-style or internal-use suggestion, you must answer in a more cautious red-tier style even if the broader discussion is educational.

Important response rules for this answer:
- You should still answer usefully and personally.
- You may discuss gentler, simpler, lower-concern herbs, teas, or formulation directions in theory if supported by the provided Green Guild material.
- Do NOT present any option as fully cleared, fully safe, or personally approved for this user.
- Prefer language such as:
  - "in theory"
  - "lower-concern conceptually"
  - "the gentler direction I'd think about"
  - "educational guidance rather than medical advice"
- Acknowledge early that the profile changes the level of caution.
- Avoid broad, confident, free-flowing recipe language.
- If you mention a blend or shortlist, keep it simple, cautious, and clearly theoretical.
- End with a short practical note telling the user to check actual use with their GP, pharmacist, or suitable clinician first.
`;
  }

  return {
    riskTier,
    flags,
    shouldForceRed,
    extraInstruction,
  };
}

async function getGuidanceDecision({ userText, userProfile, dbHistory }) {
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
- red = acute/high-risk/medically sensitive, still answer helpfully and educationally, but with stronger caution and without overconfident personal prescribing

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

This is a high-risk or medically sensitive case.
Do NOT collapse into a refusal.
Do NOT give casual, overconfident, or fully personalised prescribing language.
Do NOT present herbs as fully safe, fully cleared, or personally approved.

You should still be useful:
- acknowledge briefly that the profile changes the level of caution
- explain how an herbalist would think about the pattern
- personalise the reasoning using the user profile
- discuss gentler, simpler, lower-concern options in theory when supported by the provided material
- if relevant, you may mention a cautious shortlist, blend direction, or tea direction in theory, but keep it simple and clearly educational rather than prescriptive
- prefer wording like "in theory", "lower-concern conceptually", and "the gentler direction I'd think about"
- avoid precise dosage unless clearly safe and grounded in the provided material
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
    }

    const safetyOverride = buildSafetyOverride({
      userText,
      userProfile,
      currentRiskTier: decision.riskTier,
    });

    decision = {
      ...decision,
      riskTier: safetyOverride.riskTier,
    };

    if (userText && !imageBase64 && decision.shouldAsk && decision.question.trim()) {
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
          safetyFlags: safetyOverride.flags,
          safetyOverrideApplied: safetyOverride.shouldForceRed,
        },
      });
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
    const safetyOverrideInstruction = safetyOverride.extraInstruction
      ? `\n\n${safetyOverride.extraInstruction}`
      : "";

    const finalInstruction = `${buildSystemPrompt()}

${riskInstruction}${safetyOverrideInstruction}${formattedProfileBlock}${contextBlock}${kbFallbackBlock}`;

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
        safetyFlags: safetyOverride.flags,
        safetyOverrideApplied: safetyOverride.shouldForceRed,
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
// tools/ingest.js
// Reads raw .txt files from /kb
// Writes gzipped JSON shards to /kb_en

import fs from "fs";
import path from "path";
import zlib from "zlib";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const SRC_DIR = path.join(process.cwd(), "kb");
const OUT_DIR = path.join(process.cwd(), "kb_en");
const OUT_PREFIX = "kb_store-";

const EMB_MODEL = "gemini-embedding-001";
const CHUNK_SIZE = 900;
const CHUNK_OVERLAP = 150;
const DECIMALS = 4;
const SHARD_COUNT_TARGET = 2500;

if (!process.env.GEMINI_API_KEY) {
  console.error("Missing GEMINI_API_KEY env var.");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

function chunk(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  const step = Math.max(1, size - overlap);

  for (let i = 0; i < text.length; i += step) {
    const piece = text.slice(i, i + size).trim();
    if (piece) chunks.push(piece);
  }

  return chunks;
}

function loadTxtFiles() {
  if (!fs.existsSync(SRC_DIR)) {
    fs.mkdirSync(SRC_DIR, { recursive: true });
  }

  const files = fs.readdirSync(SRC_DIR).filter((f) => f.endsWith(".txt"));
  const docs = [];

  for (const file of files) {
    const fullPath = path.join(SRC_DIR, file);
    const full = fs.readFileSync(fullPath, "utf8");
    const parts = chunk(full);

    parts.forEach((c, i) => {
      docs.push({
        id: `${file}#${i}`,
        source: file,
        text: c,
      });
    });
  }

  return docs;
}

function roundEmbedding(arr, decimals = DECIMALS) {
  const factor = Math.pow(10, decimals);
  return arr.map((v) => Math.round(v * factor) / factor);
}

function gzipJson(obj) {
  const json = JSON.stringify(obj);
  return zlib.gzipSync(Buffer.from(json));
}

function pad(n, width = 3) {
  return String(n).padStart(width, "0");
}

function ensureOutDir() {
  if (!fs.existsSync(OUT_DIR)) {
    fs.mkdirSync(OUT_DIR, { recursive: true });
  }
}

function clearOldShards() {
  if (!fs.existsSync(OUT_DIR)) return;

  const oldFiles = fs
    .readdirSync(OUT_DIR)
    .filter((f) => f.endsWith(".json.gz"));

  for (const file of oldFiles) {
    fs.unlinkSync(path.join(OUT_DIR, file));
  }

  if (oldFiles.length) {
    console.log(`Removed ${oldFiles.length} old shard file(s) from /kb_en`);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
  const docs = loadTxtFiles();

  if (docs.length === 0) {
    console.log("No .txt files found in /kb. Add some first.");
    return;
  }

  ensureOutDir();
  clearOldShards();

  console.log(`Found ${docs.length} chunks from /kb`);
  console.log(`Embedding with ${EMB_MODEL}...`);

  const out = [];
  const BATCH = 32;

  for (let i = 0; i < docs.length; i += BATCH) {
    const batch = docs.slice(i, i + BATCH);

    try {
      const response = await ai.models.embedContent({
        model: EMB_MODEL,
        contents: batch.map((d) => d.text),
      });

      const embeddings = response.embeddings || [];

      embeddings.forEach((emb, j) => {
        const values = emb?.values || emb?.embedding?.values;
        if (!values) return;

        out.push({
          id: batch[j].id,
          source: batch[j].source,
          text: batch[j].text,
          embedding: roundEmbedding(values),
        });
      });

      console.log(`  -> ${Math.min(i + BATCH, docs.length)} / ${docs.length}`);
      await sleep(400);
    } catch (e) {
      console.error(`Error in batch starting at ${i}:`, e.message);
      await sleep(1500);
    }
  }

  if (out.length === 0) {
    console.log("No embeddings were created. Nothing to save.");
    return;
  }

  let shardIdx = 0;

  for (let start = 0; start < out.length; start += SHARD_COUNT_TARGET) {
    const end = Math.min(start + SHARD_COUNT_TARGET, out.length);
    const shard = out.slice(start, end);
    const gz = gzipJson(shard);
    const fname = `${OUT_PREFIX}${pad(shardIdx)}.json.gz`;

    fs.writeFileSync(path.join(OUT_DIR, fname), gz);

    console.log(
      `Saved ${path.join("kb_en", fname)} (${(gz.length / 1024 / 1024).toFixed(
        1
      )} MB, ${shard.length} chunks)`
    );

    shardIdx += 1;
  }

  console.log("Done. English KB shards are ready in /kb_en");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
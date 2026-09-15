/**
 * GitHub Pages 정적 프로토타입 클라이언트
 * ------------------------------------------------
 * 서버가 없으므로:
 *  - 법령 청크 데이터(data/chunks.json)를 브라우저에서 직접 로드
 *  - TF-IDF 기반 경량 키워드 검색으로 관련 조문 검색 (구조적 청킹 데이터 활용)
 *  - 검색된 근거를 프롬프트에 주입하여 OpenAI / Google Gemini / Anthropic Claude 중 선택된 제공사의 API를 브라우저에서 직접 호출
 *  - 지역 법률 인프라 자동 매핑 로직 포함
 *
 * 주의: API 키는 제공사별로 localStorage에만 저장되고 브라우저에서 해당 제공사 API로 직접 전송됩니다.
 *       (진짜 서비스라면 백엔드 프록시를 두어 키를 숨겨야 합니다.)
 */

const INFRA_DIRECTORY = [
  {
    keywords: ["보조금", "산업단지", "중소기업", "스타트업", "지원"],
    name: "광주테크노파크 법률지원단",
    desc: "산업단지 입주규정·보조금 관련 법률 자문",
    contact: "062-123-4567",
    link: "https://www.gjtp.or.kr",
  },
  {
    keywords: ["근로", "노무", "해고", "임금", "휴가"],
    name: "광주지방고용노동청",
    desc: "근로기준법 위반 신고 및 노무 상담",
    contact: "1350 (고용노동상담센터)",
    link: "https://www.moel.go.kr",
  },
  {
    keywords: ["nda", "비밀유지", "계약서", "독소조항"],
    name: "광주광역시 무료 법률상담소",
    desc: "계약서 검토 및 독소조항 관련 무료 법률상담",
    contact: "062-613-2000",
    link: "https://www.gwangju.go.kr/law_counsel",
  },
  {
    keywords: ["전남", "지방보조금", "도지사"],
    name: "전남테크노파크 기업지원센터",
    desc: "전남 지역 보조금·기업지원 규정 상담",
    contact: "061-729-3114",
    link: "https://www.jntp.or.kr",
  },
];

const DEFAULT_INFRA = [
  {
    name: "광주테크노파크 법률지원단",
    desc: "지역 스타트업/중소기업 법률 자문 전반",
    contact: "062-123-4567",
    link: "https://www.gjtp.or.kr",
  },
];

const SYSTEM_TEMPLATE = (context, question) => `
당신은 광주/전남 지역의 스타트업 및 중소기업을 위한 B2B 맞춤형 법률·규정 검토 전문 AI 에이전트입니다.
반드시 [참고 법령/규정]에 제시된 내용만을 근거로 답변하세요. 근거에 없는 내용은 "제공된 자료에서 확인되지 않음"이라고 명시하고 추측하지 마세요. (환각 현상 방지)

[참고 법령/규정]
${context}

[사용자 질의]
${question}

[답변 형식]
1. 규정 스캐닝 결과: 위반 소지 및 독소조항 여부 (있음/없음/판단불가)
2. 상세 근거: 조·항·호 번호를 명시하여 설명 (예: OO조례 제5조 제2항)
3. 실무 권고사항: 사용자가 다음에 취해야 할 조치
`;

// 2026-09 기준 실제 사용 가능한(검증된) 모델만 등록
const PROVIDER_MODELS = {
  openai: ["gpt-5.5", "gpt-5.1", "gpt-4o-mini"],
  gemini: ["gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
  claude: ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
};

const PROVIDER_KEY_LABEL = {
  openai: "OpenAI API Key",
  gemini: "Google AI Studio (Gemini) API Key",
  claude: "Anthropic (Claude) API Key",
};

let CHUNKS = [];

async function loadChunks() {
  const res = await fetch("data/chunks.json");
  CHUNKS = await res.json();
}

// ----------------------------------------------------------------
// 경량 TF-IDF 키워드 검색 (한국어 형태소 분석기 없이 어절 단위로 근사)
// ----------------------------------------------------------------
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

function buildIndex(chunks) {
  const df = new Map();
  const docs = chunks.map((c) => {
    const tokens = tokenize(c.text);
    const tf = new Map();
    tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    new Set(tokens).forEach((t) => df.set(t, (df.get(t) || 0) + 1));
    return { chunk: c, tf, length: tokens.length };
  });
  return { docs, df, N: chunks.length };
}

function scoreQuery(query, index) {
  const qTokens = tokenize(query);
  const scores = index.docs.map(({ chunk, tf, length }) => {
    let score = 0;
    qTokens.forEach((qt) => {
      const tfVal = tf.get(qt) || 0;
      if (tfVal === 0) return;
      const df = index.df.get(qt) || 1;
      const idf = Math.log((index.N + 1) / df) + 1;
      score += (tfVal / (length || 1)) * idf;
    });
    // 조 번호/법령명이 질의에 등장하면 가산점 (키워드 검색 보강)
    if (query.includes(chunk.article_no)) score += 0.5;
    return { chunk, score };
  });
  return scores.sort((a, b) => b.score - a.score);
}

function search(query, region, k) {
  const index = buildIndex(CHUNKS);
  let ranked = scoreQuery(query, index);
  if (region && region !== "전체") {
    ranked = ranked.filter(
      (r) => r.chunk.region === region || r.chunk.region === "전국"
    );
  }
  return ranked.slice(0, k).map((r) => r.chunk);
}

function formatContext(chunks) {
  return chunks
    .map((c) => `(${c.law_name} ${c.article_no}) ${c.text}`)
    .join("\n\n");
}

function recommendInfra(query, categories) {
  const haystack = (query + " " + categories.join(" ")).toLowerCase();
  let matched = INFRA_DIRECTORY.filter((infra) =>
    infra.keywords.some((kw) => haystack.includes(kw))
  );
  if (matched.length === 0) matched = DEFAULT_INFRA;
  const seen = new Set();
  return matched.filter((m) => {
    if (seen.has(m.name)) return false;
    seen.add(m.name);
    return true;
  });
}

function formatInfraMarkdown(infraList) {
  const lines = ["\n💡 지역 법률 인프라 연계"];
  infraList.forEach((infra) => {
    lines.push(`- ${infra.name} (${infra.desc}) · 연락처: ${infra.contact} · ${infra.link}`);
  });
  return lines.join("\n");
}

// ----------------------------------------------------------------
// 제공사별 LLM 호출 (모두 브라우저에서 직접 CORS 호출)
// ----------------------------------------------------------------
async function callOpenAI(apiKey, model, prompt) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`OpenAI API 오류 (${res.status}): ${errBody}`);
  }
  const data = await res.json();
  return data.choices[0].message.content;
}

// Gemini generateContent REST API: 별도 CORS 헤더 없이 브라우저에서 직접 호출 가능
async function callGemini(apiKey, model, prompt) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0 },
    }),
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`Gemini API 오류 (${res.status}): ${errBody}`);
  }
  const data = await res.json();
  return data.candidates[0].content.parts.map((p) => p.text).join("");
}

// Anthropic Messages API: anthropic-dangerous-direct-browser-access 헤더로 브라우저 직접 호출 허용
async function callClaude(apiKey, model, prompt) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model,
      max_tokens: 2048,
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`Claude API 오류 (${res.status}): ${errBody}`);
  }
  const data = await res.json();
  return data.content.map((c) => c.text).join("");
}

async function callLLM(provider, apiKey, model, prompt) {
  if (provider === "openai") return callOpenAI(apiKey, model, prompt);
  if (provider === "gemini") return callGemini(apiKey, model, prompt);
  if (provider === "claude") return callClaude(apiKey, model, prompt);
  throw new Error(`알 수 없는 제공사: ${provider}`);
}

// ----------------------------------------------------------------
// UI 로직
// ----------------------------------------------------------------
const chatEl = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("userQuery");
const apiKeyInput = document.getElementById("apiKey");
const apiKeyLabel = document.getElementById("apiKeyLabel");
const providerSelect = document.getElementById("providerSelect");
const regionFilter = document.getElementById("regionFilter");
const topKInput = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const modelName = document.getElementById("modelName");
const statusMsg = document.getElementById("statusMsg");

function currentProvider() {
  return providerSelect.value;
}

function refreshModelOptions() {
  const provider = currentProvider();
  modelName.innerHTML = "";
  PROVIDER_MODELS[provider].forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelName.appendChild(opt);
  });
  apiKeyLabel.textContent = PROVIDER_KEY_LABEL[provider];
  apiKeyInput.placeholder = provider === "openai" ? "sk-..." : provider === "gemini" ? "AIza..." : "sk-ant-...";
  apiKeyInput.value = localStorage.getItem(`${provider}_api_key`) || "";
}

providerSelect.addEventListener("change", refreshModelOptions);
refreshModelOptions();

apiKeyInput.addEventListener("change", () => {
  localStorage.setItem(`${currentProvider()}_api_key`, apiKeyInput.value);
});
topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const apiKey = apiKeyInput.value.trim();
  const question = input.value.trim();
  if (!question) return;

  if (!apiKey) {
    statusMsg.textContent = `👈 왼쪽 패널에 ${PROVIDER_KEY_LABEL[currentProvider()]}를 입력해야 서비스를 시작할 수 있습니다.`;
    return;
  }
  statusMsg.textContent = "";

  appendMessage("user", question);
  input.value = "";

  const region = regionFilter.value;
  const k = parseInt(topKInput.value, 10);
  const model = modelName.value;
  const provider = currentProvider();

  const loadingDiv = appendMessage("assistant", "관련 광주/전남 조례 및 규정을 검색 중입니다...");

  try {
    const docs = search(question, region, k);
    const context = formatContext(docs);
    const prompt = SYSTEM_TEMPLATE(context, question);
    const answer = await callLLM(provider, apiKey, model, prompt);
    const categories = docs.map((d) => d.category || "");
    const infra = recommendInfra(question, categories);
    loadingDiv.textContent = answer + "\n\n" + formatInfraMarkdown(infra);
  } catch (err) {
    loadingDiv.textContent = `오류가 발생했습니다: ${err.message}`;
  }
});

loadChunks();

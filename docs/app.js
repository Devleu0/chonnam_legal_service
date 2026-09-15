/**
 * GitHub Pages 정적 프로토타입 클라이언트
 * ------------------------------------------------
 * 서버가 없으므로:
 *  - 법령 청크 데이터(data/chunks.json)를 브라우저에서 직접 로드
 *  - TF-IDF 기반 경량 키워드 검색으로 관련 조문 검색 (구조적 청킹 데이터 활용)
 *  - 검색된 근거를 프롬프트에 주입하여 OpenAI Chat Completions API를 브라우저에서 직접 호출
 *  - 지역 법률 인프라 자동 매핑 로직 포함
 *
 * 주의: API 키는 localStorage에만 저장되고 브라우저에서 OpenAI로 직접 전송됩니다.
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
// OpenAI 호출
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

// ----------------------------------------------------------------
// UI 로직
// ----------------------------------------------------------------
const chatEl = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("userQuery");
const apiKeyInput = document.getElementById("apiKey");
const regionFilter = document.getElementById("regionFilter");
const topKInput = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const modelName = document.getElementById("modelName");
const statusMsg = document.getElementById("statusMsg");

apiKeyInput.value = localStorage.getItem("openai_api_key") || "";
apiKeyInput.addEventListener("change", () => {
  localStorage.setItem("openai_api_key", apiKeyInput.value);
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
    statusMsg.textContent = "👈 왼쪽 패널에 OpenAI API Key를 입력해야 서비스를 시작할 수 있습니다.";
    return;
  }
  statusMsg.textContent = "";

  appendMessage("user", question);
  input.value = "";

  const region = regionFilter.value;
  const k = parseInt(topKInput.value, 10);
  const model = modelName.value;

  const loadingDiv = appendMessage("assistant", "관련 광주/전남 조례 및 규정을 검색 중입니다...");

  try {
    const docs = search(question, region, k);
    const context = formatContext(docs);
    const prompt = SYSTEM_TEMPLATE(context, question);
    const answer = await callOpenAI(apiKey, model, prompt);
    const categories = docs.map((d) => d.category || "");
    const infra = recommendInfra(question, categories);
    loadingDiv.textContent = answer + "\n\n" + formatInfraMarkdown(infra);
  } catch (err) {
    loadingDiv.textContent = `오류가 발생했습니다: ${err.message}`;
  }
});

loadChunks();

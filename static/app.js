let selectedPerson = 0;
let personLocked = false;

const $ = (id) => document.getElementById(id);
const FRONTEND_PLAYER_SCOPE = "paris_2024_usa";
const VISIBLE_PLAYER_KEYS = new Set([
  "stephen_curry",
  "devin_booker",
  "kevin_durant",
  "anthony_edwards",
  "lebron_james",
]);
const JOINTS = ["elbow", "shoulder", "hip", "knee"];
const PHASES = [
  { id: "catch", label: "캐치" },
  { id: "dip", label: "딥" },
  { id: "rise", label: "라이즈" },
  { id: "release", label: "릴리즈" },
  { id: "follow_through", label: "팔로우스루" },
];
const JOINT_COLORS = {
  elbow: "#ea580c",
  shoulder: "#16a34a",
  hip: "#1e3a5f",
  knee: "#7c3aed",
};

function createSkeletonViewer(prefix) {
  return new window.Skeleton3D.SkeletonViewer({
    canvas: $(`${prefix}_skeleton_canvas`),
    playButton: $(`${prefix}_skeleton_play`),
    slider: $(`${prefix}_skeleton_slider`),
    timeLabel: $(`${prefix}_skeleton_time`),
    phaseLabel: $(`${prefix}_skeleton_phase`),
  });
}

const userSkeletonViewer = createSkeletonViewer("user");
const playerSkeletonViewer = createSkeletonViewer("player");

function skeletonMeta(profile, fallbackView = "side") {
  const frames = profile?.frames?.length || 0;
  const landmarks = Object.keys(profile?.frames?.[0]?.landmarks || {}).length;
  const quality = profile?.quality_mode === "multi_view_3d"
    ? "Multi-view 3D"
    : "Single-view estimated 3D";
  const view = String(profile?.view || fallbackView).toUpperCase();
  return `${quality} · ${frames}프레임 · ${landmarks}랜드마크 · ${view}`;
}

async function loadPlayerSkeleton(playerKey, displayName, scroll = true) {
  const section = $("player_skeleton_section");
  section.hidden = false;
  $("player_skeleton_title").textContent = `${displayName} · 3D 스켈레톤`;
  $("player_skeleton_meta").textContent = "3D 스켈레톤을 불러오는 중…";
  try {
    const res = await fetch(`/api/players/${encodeURIComponent(playerKey)}/skeleton`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "선수 스켈레톤 로드 실패");
    playerSkeletonViewer.setProfile(data.skeleton);
    $("player_skeleton_meta").textContent = skeletonMeta(data.skeleton);
    if (scroll) section.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) {
    $("player_skeleton_meta").textContent = String(err.message || err);
  }
}

function showUserSkeleton(views) {
  const candidates = (views || []).filter((view) => view.skeleton?.frames?.length);
  const view = candidates.find((item) => item.view === "side") || candidates[0];
  const section = $("user_skeleton_section");
  section.hidden = !view;
  if (!view) return;
  userSkeletonViewer.setProfile(view.skeleton);
  $("user_skeleton_meta").textContent = skeletonMeta(view.skeleton, view.view);
}

function firstVideoFile() {
  return $("video_side").files[0] || $("video_front").files[0] || $("video_oblique").files[0] || null;
}

function appendVideos(fd) {
  const side = $("video_side").files[0];
  const front = $("video_front").files[0];
  const oblique = $("video_oblique").files[0];
  if (side) fd.append("video_side", side);
  if (front) fd.append("video_front", front);
  if (oblique) fd.append("video_oblique", oblique);
  return Boolean(side || front || oblique);
}

function setStep(n) {
  document.querySelectorAll(".step").forEach((el) => {
    const step = Number(el.dataset.step);
    el.classList.toggle("is-active", step === n);
    el.classList.toggle("is-done", step < n);
  });
}

function setBusy(on, text) {
  $("status").textContent = text || "";
  $("progress").hidden = !on;
  $("btn_analyze").disabled = on || !firstVideoFile();
  $("btn_people").disabled = on;
}

function bindFileLabel(inputId, nameId) {
  $(inputId).addEventListener("change", () => {
    const f = $(inputId).files[0];
    $(nameId).textContent = f ? f.name : "파일 선택";
    $("btn_analyze").disabled = !firstVideoFile();
    if (firstVideoFile()) setStep(1);
  });
}

bindFileLabel("video_side", "name_side");
bindFileLabel("video_front", "name_front");
bindFileLabel("video_oblique", "name_oblique");

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const el = $("health");
    if (data.ok) {
      el.textContent = `API OK · ${VISIBLE_PLAYER_KEYS.size} Olympic profiles`;
      el.classList.add("ok");
    } else {
      el.textContent = "API unavailable";
    }
  } catch {
    $("health").textContent = "API offline";
  }
}

async function loadPlayers() {
  const res = await fetch("/api/players?scope=paris_2024_usa");
  const data = await res.json();
  const root = $("players");
  const selector = $("target_player");
  root.innerHTML = "";
  selector.length = 1;
  const visiblePlayers = (data.players || []).filter((p) => VISIBLE_PLAYER_KEYS.has(p.player_key));
  visiblePlayers.forEach((p) => {
    const option = document.createElement("option");
    option.value = p.player_key;
    option.textContent = p.display_name;
    selector.appendChild(option);

    const a = p.angles || {};
    const el = document.createElement("div");
    el.className = "match";
    el.innerHTML = `
      <div>
        <strong>${p.display_name}</strong>
        <div><em>3D 관절각 프로필</em></div>
      </div>
      <em>E ${Number(a.elbow).toFixed(1)}°</em>
      <em>S ${Number(a.shoulder).toFixed(1)}°</em>
      <button type="button" class="btn ghost compact skeleton-open">3D 스켈레톤</button>
    `;
    el.querySelector(".skeleton-open").addEventListener("click", () => {
      loadPlayerSkeleton(p.player_key, p.display_name);
    });
    root.appendChild(el);
  });
  if (visiblePlayers.length) {
    const first = visiblePlayers[0];
    await loadPlayerSkeleton(first.player_key, first.display_name, false);
  }
}

$("btn_people").addEventListener("click", async () => {
  const file = firstVideoFile();
  if (!file) {
    $("status").textContent = "영상을 하나 이상 선택하세요.";
    return;
  }
  setBusy(true, "사람 탐지 중…");
  try {
    const fd = new FormData();
    fd.append("video", file);
    const res = await fetch("/api/people", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "사람 탐지 실패");

    const list = $("people_list");
    list.innerHTML = "";
    (data.people || []).forEach((p, idx) => {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "person";
      el.innerHTML = `<img alt="person ${p.person_index}" src="data:image/jpeg;base64,${p.thumb_jpeg_b64}" /><span>#${p.person_index}</span>`;
      el.addEventListener("click", () => {
        selectedPerson = p.person_index;
        personLocked = true;
        [...list.children].forEach((c) => c.classList.remove("selected"));
        el.classList.add("selected");
      });
      list.appendChild(el);
      if (idx === 0) el.classList.add("selected");
    });
    $("people_section").hidden = false;
    setStep(2);
    if (data.people?.length) {
      selectedPerson = data.people[0].person_index;
      $("status").textContent = `${data.people.length}명 감지. 분석할 사람을 선택하세요.`;
    } else {
      $("status").textContent = "사람을 찾지 못했습니다. 밝은 전신 영상을 사용해 보세요.";
    }
  } catch (err) {
    $("status").textContent = String(err.message || err);
  } finally {
    setBusy(false);
  }
});

function renderResult(data) {
  $("result_section").hidden = false;
  $("result_json").textContent = JSON.stringify(data, null, 2);
  $("disclaimer").textContent = data.disclaimer || "";
  setStep(3);

  const angles = data.analysis?.release_angles_merged || {};
  const metrics = $("metrics");
  metrics.innerHTML = "";
  JOINTS.forEach((k) => {
    const el = document.createElement("div");
    el.className = "metric";
    el.innerHTML = `<span>${k}</span><b>${Number(angles[k] ?? 0).toFixed(1)}°</b>`;
    metrics.appendChild(el);
  });

  const top = data.closest_match || (data.matches || [])[0];
  $("top_name").textContent = top ? top.display_name : "—";
  $("top_score").textContent = top ? top.score.toFixed(1) : "—";
  $("top_dist").textContent = top ? `${top.distance_deg.toFixed(1)}°` : "—";

  const selected = data.selected_match;
  $("selected_compare").hidden = !selected;
  if (selected) {
    $("selected_name").textContent = selected.display_name;
    $("selected_score").textContent = selected.score.toFixed(1);
    $("selected_dist").textContent = `${selected.distance_deg.toFixed(1)}°`;
  }
  $("feedback_title").textContent = selected
    ? `코칭 포인트 · ${selected.display_name} 기준`
    : "코칭 포인트 · 최유사 선수 기준";

  const fb = $("feedback");
  fb.innerHTML = "";
  (data.feedback || []).forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    fb.appendChild(li);
  });

  const matches = $("matches");
  matches.innerHTML = "";
  (data.matches || []).forEach((m) => {
    const el = document.createElement("div");
    el.className = "match";
    el.innerHTML = `
      <div><strong>${m.display_name}</strong><div><em>3D${m.matched_view ? " · " + m.matched_view : ""}</em></div></div>
      <em>${m.score.toFixed(1)} pts</em>
      <em>Δ ${m.distance_deg.toFixed(1)}°</em>
    `;
    matches.appendChild(el);
  });

  renderPhases(data.analysis?.views || []);
  showUserSkeleton(data.analysis?.views || []);

  const reference = selected || top;
  if (reference) loadPlayerSkeleton(reference.player_key, reference.display_name, false);

  $("result_section").scrollIntoView({ behavior: "smooth", block: "start" });
}

function fmtTime(t) {
  if (t == null || Number.isNaN(Number(t))) return "—";
  return `${Number(t).toFixed(2)}s`;
}

function fmtDeg(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Number(v).toFixed(1)}°`;
}

function renderPhases(views) {
  const root = $("phase_views");
  root.innerHTML = "";
  const usable = (views || []).filter((v) => (v.timeline || []).length);
  if (!usable.length) {
    root.innerHTML = "<p class='phase-lead'>이 클립에서는 슈팅 구간을 나누지 못했습니다. 전신이 보이는 측면 영상을 올려 보세요.</p>";
    return;
  }
  usable.forEach((view) => {
    const wrap = document.createElement("div");
    wrap.className = "phase-view";
    const summary = view.phase_summary || [];
    const samples = view.timeline || [];
    const lastT = samples.length ? samples[samples.length - 1].t : 0;

    const chips = PHASES.map((p) => {
      const row = summary.find((s) => s.phase === p.id);
      const count = row?.count || 0;
      const span = count ? `${fmtTime(row.t_start)}–${fmtTime(row.t_end)} · ${count}f` : "없음";
      return `<button type="button" class="phase-chip" data-phase="${p.id}"><b>${p.label}</b><span>${span}</span></button>`;
    }).join("");

    const rows = PHASES.map((p) => {
      const row = summary.find((s) => s.phase === p.id);
      const a = row?.angles || {};
      return `<tr data-phase="${p.id}"><td>${p.label}</td><td>${row?.count || 0}</td><td>${fmtTime(row?.t_start)}</td><td>${fmtDeg(a.elbow)}</td><td>${fmtDeg(a.shoulder)}</td><td>${fmtDeg(a.hip)}</td><td>${fmtDeg(a.knee)}</td></tr>`;
    }).join("");

    wrap.innerHTML = `
      <h4>${String(view.view || "view").toUpperCase()} · ${samples.length} frames · ${fmtTime(lastT)}</h4>
      <div class="phase-bar">${chips}</div>
      <table class="phase-table">
        <thead><tr><th>구간</th><th>프레임</th><th>시작</th><th>팔꿈치</th><th>어깨</th><th>골반</th><th>무릎</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${drawPhaseChart(samples, summary)}
      <p class="phase-legend">
        <span><i style="background:${JOINT_COLORS.elbow}"></i>팔꿈치</span>
        <span><i style="background:${JOINT_COLORS.shoulder}"></i>어깨</span>
        <span><i style="background:${JOINT_COLORS.hip}"></i>골반</span>
        <span><i style="background:${JOINT_COLORS.knee}"></i>무릎</span>
      </p>
    `;
    wrap.querySelectorAll(".phase-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const phase = btn.dataset.phase;
        wrap.querySelectorAll(".phase-chip").forEach((c) => c.classList.toggle("is-on", c === btn));
        wrap.querySelectorAll(".phase-table tr").forEach((tr) => tr.classList.toggle("is-on", tr.dataset.phase === phase));
      });
    });
    root.appendChild(wrap);
  });
}

function drawPhaseChart(samples, summary) {
  if (!samples.length) return "";
  const w = 640;
  const h = 168;
  const pad = { l: 36, r: 10, t: 10, b: 24 };
  const tMax = Math.max(samples[samples.length - 1].t, 0.001);
  const x = (t) => pad.l + ((t / tMax) * (w - pad.l - pad.r));
  const y = (deg) => pad.t + ((180 - Math.min(180, Math.max(0, deg))) / 180) * (h - pad.t - pad.b);

  const bands = summary
    .filter((s) => s.count)
    .map((s, i) => {
      const x0 = x(s.t_start);
      const x1 = x(s.t_end);
      const fill = i % 2 === 0 ? "rgba(249,115,22,0.08)" : "rgba(22,163,74,0.08)";
      return `<rect x="${x0}" y="${pad.t}" width="${Math.max(1, x1 - x0)}" height="${h - pad.t - pad.b}" fill="${fill}"></rect>`;
    })
    .join("");

  const lines = JOINTS.map((joint) => {
    const pts = samples.map((s) => `${x(s.t).toFixed(1)},${y(s[joint]).toFixed(1)}`).join(" ");
    return `<polyline fill="none" stroke="${JOINT_COLORS[joint]}" stroke-width="2" points="${pts}"></polyline>`;
  }).join("");

  return `<svg class="phase-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="시간별 관절 각도">${bands}${lines}</svg>`;
}

$("btn_analyze").addEventListener("click", async () => {
  const fd = new FormData();
  if (!appendVideos(fd)) {
    $("status").textContent = "영상을 하나 이상 선택하세요.";
    return;
  }
  fd.append("person_index", String(selectedPerson));
  fd.append("auto_person", personLocked ? "0" : "1");
  fd.append("lang", "ko");
  fd.append("catalog_scope", FRONTEND_PLAYER_SCOPE);
  const hand = $("hand").value;
  if (hand) fd.append("hand", hand);
  const targetPlayer = $("target_player").value;
  if (targetPlayer) fd.append("target_player_key", targetPlayer);

  setBusy(true, "3D 각도 분석 중… (1~2분 걸릴 수 있습니다)");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "분석 실패");
    renderResult(data);
    $("status").textContent = targetPlayer
      ? "완료 — 선택 선수 비교와 최유사 선수를 함께 찾았습니다."
      : "완료 — 가장 유사한 선수를 찾았습니다.";
  } catch (err) {
    $("status").textContent = String(err.message || err);
  } finally {
    setBusy(false);
  }
});

loadHealth();
loadPlayers();
setStep(1);

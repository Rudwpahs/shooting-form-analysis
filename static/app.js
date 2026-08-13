let selectedPerson = 0;

const $ = (id) => document.getElementById(id);
const JOINTS = ["elbow", "shoulder", "hip", "knee"];

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
      el.textContent = `API OK · ${data.players} profiles`;
      el.classList.add("ok");
    } else {
      el.textContent = "API unavailable";
    }
  } catch {
    $("health").textContent = "API offline";
  }
}

async function loadPlayers() {
  const res = await fetch("/api/players");
  const data = await res.json();
  const root = $("players");
  root.innerHTML = "";
  (data.players || []).forEach((p) => {
    const a = p.angles || {};
    const el = document.createElement("div");
    el.className = "match";
    el.innerHTML = `
      <div>
        <strong>${p.display_name}</strong>
        <div><em>${p.source || ""} · ${p.space || ""}</em></div>
      </div>
      <em>E ${Number(a.elbow).toFixed(1)}°</em>
      <em>S ${Number(a.shoulder).toFixed(1)}°</em>
    `;
    root.appendChild(el);
  });
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

  const top = (data.matches || [])[0];
  $("top_name").textContent = top ? top.display_name : "—";
  $("top_score").textContent = top ? top.score.toFixed(1) : "—";
  $("top_dist").textContent = top ? `${top.distance_deg.toFixed(1)}°` : "—";

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
      <div><strong>${m.display_name}</strong><div><em>${m.player_key}</em></div></div>
      <em>${m.score.toFixed(1)} pts</em>
      <em>Δ ${m.distance_deg.toFixed(1)}°</em>
    `;
    matches.appendChild(el);
  });

  $("result_section").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("btn_analyze").addEventListener("click", async () => {
  const fd = new FormData();
  if (!appendVideos(fd)) {
    $("status").textContent = "영상을 하나 이상 선택하세요.";
    return;
  }
  fd.append("person_index", String(selectedPerson));
  fd.append("lang", "ko");
  const hand = $("hand").value;
  if (hand) fd.append("hand", hand);

  setBusy(true, "3D 각도 분석 중… (1~2분 걸릴 수 있습니다)");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "분석 실패");
    renderResult(data);
    $("status").textContent = "완료 — 각도(°)만으로 매칭했습니다.";
  } catch (err) {
    $("status").textContent = String(err.message || err);
  } finally {
    setBusy(false);
  }
});

loadHealth();
loadPlayers();
setStep(1);

"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

type Language = "ko" | "en";
type Hand = "right" | "left";
type MetricKey = "elbow" | "shoulder" | "hip" | "knee";

type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
};

type Analysis = {
  metrics: Record<MetricKey, number>;
  score: number;
  releaseTime: number;
  landmarks: Landmark[];
  feedback: string[];
};

const copy = {
  ko: {
    eyebrow: "AI-POWERED BIOMECHANICS",
    heroA: "당신의 슛을",
    heroB: "눈으로 확인하세요.",
    sub: "영상 하나면 충분합니다. 릴리스 순간의 자세를 추적하고, 프로 기준과 비교해 바로 적용할 수 있는 교정 포인트를 알려드려요.",
    private: "영상은 브라우저 안에서만 처리됩니다",
    start: "분석 시작",
    how: "분석 원리",
    uploadTitle: "슈팅 영상을 올려주세요",
    uploadSub: "측면에서 전신이 보이는 5–15초 영상을 권장합니다",
    browse: "영상 선택",
    replace: "다른 영상 선택",
    settings: "분석 설정",
    hand: "슈팅 핸드",
    right: "오른손",
    left: "왼손",
    reference: "비교 기준",
    curry: "Stephen Curry",
    baseline: "프로 밸런스 기준",
    modelNote: "저장소 내 실제 모델 · 샘플 1개",
    baselineNote: "릴리스 역학 권장 범위",
    analyze: "내 슛 분석하기",
    analyzing: "릴리스 프레임을 찾는 중",
    modelLoading: "포즈 모델 준비 → 프레임 추적 → 각도 계산",
    score: "FORM SCORE",
    release: "릴리스 프레임",
    comparison: "각도 비교",
    yours: "나의 각도",
    target: "기준",
    delta: "차이",
    coaching: "코칭 포인트",
    again: "새 영상 분석",
    elbow: "팔꿈치",
    shoulder: "어깨",
    hip: "골반",
    knee: "무릎",
    step1: "01 / 추적",
    step1Title: "33개 포즈 랜드마크",
    step1Body: "MediaPipe가 영상 속 관절 움직임을 프레임 단위로 읽습니다.",
    step2: "02 / 감지",
    step2Title: "릴리스 순간 자동 탐색",
    step2Body: "손목 높이와 팔꿈치 신전을 함께 계산해 핵심 프레임을 찾습니다.",
    step3: "03 / 코칭",
    step3Title: "바로 쓰는 교정 포인트",
    step3Body: "프로 기준과의 각도 차이를 우선순위가 높은 피드백으로 바꿉니다.",
    footer: "더 나은 슛은 더 선명한 피드백에서 시작됩니다.",
    errorNoPose: "선수의 전신 포즈를 충분히 찾지 못했습니다. 밝은 곳에서 측면 전신이 보이는 영상으로 다시 시도해 주세요.",
    errorGeneric: "분석 모델을 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.",
  },
  en: {
    eyebrow: "AI-POWERED BIOMECHANICS",
    heroA: "See your shot.",
    heroB: "Fix your form.",
    sub: "One clip is enough. Track your release mechanics, compare them with a pro reference, and get coaching cues you can use on your next rep.",
    private: "Your video never leaves this browser",
    start: "Start analysis",
    how: "How it works",
    uploadTitle: "Drop in your shooting clip",
    uploadSub: "A 5–15 second, full-body side view works best",
    browse: "Choose video",
    replace: "Choose another",
    settings: "Analysis setup",
    hand: "Shooting hand",
    right: "Right",
    left: "Left",
    reference: "Reference",
    curry: "Stephen Curry",
    baseline: "Pro balance baseline",
    modelNote: "Repository model · 1 sample",
    baselineNote: "Recommended release mechanics",
    analyze: "Analyze my shot",
    analyzing: "Finding your release frame",
    modelLoading: "Loading pose model → tracking frames → measuring angles",
    score: "FORM SCORE",
    release: "Release frame",
    comparison: "Angle comparison",
    yours: "Your angle",
    target: "Target",
    delta: "Delta",
    coaching: "Coaching cues",
    again: "Analyze a new clip",
    elbow: "Elbow",
    shoulder: "Shoulder",
    hip: "Hip",
    knee: "Knee",
    step1: "01 / TRACK",
    step1Title: "33 pose landmarks",
    step1Body: "MediaPipe reads joint movement through your clip, frame by frame.",
    step2: "02 / DETECT",
    step2Title: "Automatic release sync",
    step2Body: "Wrist height and elbow extension reveal the most important frame.",
    step3: "03 / COACH",
    step3Title: "Cues for your next rep",
    step3Body: "Angle deltas become a short, prioritized list of adjustments.",
    footer: "Better shots start with clearer feedback.",
    errorNoPose: "We could not track a clear full-body pose. Try a brighter side-view clip with your full body in frame.",
    errorGeneric: "The analysis model could not load. Check your connection and try again.",
  },
} as const;

const references = {
  curry: {
    elbow: 179.6,
    shoulder: 152.7,
    hip: 175.9,
    knee: 177.2,
  },
  baseline: {
    elbow: 170,
    shoulder: 145,
    hip: 168,
    knee: 165,
  },
} satisfies Record<string, Record<MetricKey, number>>;

const metricKeys: MetricKey[] = ["elbow", "shoulder", "hip", "knee"];

function angle(a: Landmark, b: Landmark, c: Landmark) {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const dot = ba.x * bc.x + ba.y * bc.y;
  const mag = Math.hypot(ba.x, ba.y) * Math.hypot(bc.x, bc.y);
  if (!mag) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

function seek(video: HTMLVideoElement, time: number) {
  return new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error("seek timeout")), 5000);
    const done = () => {
      window.clearTimeout(timer);
      resolve();
    };
    video.addEventListener("seeked", done, { once: true });
    video.currentTime = time;
  });
}

function buildFeedback(
  metrics: Record<MetricKey, number>,
  target: Record<MetricKey, number>,
  lang: Language,
) {
  const rules: Record<MetricKey, [string, string]> = {
    elbow: [
      "릴리스 때 팔꿈치를 조금 더 펴서 공에 에너지를 끝까지 전달해 보세요.",
      "Extend through the elbow a little longer to finish the energy transfer.",
    ],
    shoulder: [
      "공을 이마 앞에서 놓을 수 있도록 슈팅 포켓을 조금 높여보세요.",
      "Raise the shooting pocket slightly so the ball releases in front of your forehead.",
    ],
    hip: [
      "상체가 접히지 않게 골반과 가슴을 림 방향으로 길게 세워보세요.",
      "Stay tall through the hips and chest instead of folding at release.",
    ],
    knee: [
      "릴리스 순간까지 무릎 드라이브를 이어가며 위로 힘을 연결해 보세요.",
      "Carry your knee drive into the release to connect force upward.",
    ],
  };
  return metricKeys
    .map((key) => ({ key, diff: target[key] - metrics[key] }))
    .sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff))
    .filter(({ diff }) => Math.abs(diff) > 6)
    .slice(0, 3)
    .map(({ key }) => rules[key][lang === "ko" ? 0 : 1])
    .concat(
      metricKeys.every((key) => Math.abs(target[key] - metrics[key]) <= 6)
        ? [
            lang === "ko"
              ? "릴리스 밸런스가 기준 범위에 들어왔습니다. 같은 리듬을 반복해 보세요."
              : "Your release is inside the target range. Repeat the same rhythm.",
          ]
        : [],
    );
}

export default function Home() {
  const [lang, setLang] = useState<Language>("ko");
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [hand, setHand] = useState<Hand>("right");
  const [reference, setReference] = useState<keyof typeof references>("curry");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const t = copy[lang];

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    if (!analysis || !videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const draw = async () => {
      await seek(video, analysis.releaseTime).catch(() => undefined);
      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, width, height);
      ctx.strokeStyle = "#c8ff3d";
      ctx.fillStyle = "#c8ff3d";
      ctx.lineWidth = Math.max(4, width / 220);
      const side = hand === "right" ? [12, 14, 16, 24, 26, 28] : [11, 13, 15, 23, 25, 27];
      const pairs = [
        [side[2], side[1]],
        [side[1], side[0]],
        [side[0], side[3]],
        [side[3], side[4]],
        [side[4], side[5]],
      ];
      pairs.forEach(([a, b]) => {
        const p1 = analysis.landmarks[a];
        const p2 = analysis.landmarks[b];
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      });
      side.forEach((index) => {
        const point = analysis.landmarks[index];
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, Math.max(6, width / 150), 0, Math.PI * 2);
        ctx.fill();
      });
    };
    draw();
  }, [analysis, hand]);

  const acceptFile = (next: File | undefined) => {
    if (!next || !next.type.startsWith("video/")) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(next);
    setVideoUrl(URL.createObjectURL(next));
    setAnalysis(null);
    setError("");
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  };

  const analyze = async () => {
    const video = videoRef.current;
    if (!video || !file) return;
    setLoading(true);
    setProgress(4);
    setError("");
    setAnalysis(null);
    try {
      const { FilesetResolver, PoseLandmarker } = await import("@mediapipe/tasks-vision");
      const wasm = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm",
      );
      setProgress(16);
      let landmarker;
      try {
        landmarker = await PoseLandmarker.createFromOptions(wasm, {
          baseOptions: { modelAssetPath: "/models/pose_landmarker_full.task", delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.45,
          minPosePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
      } catch {
        landmarker = await PoseLandmarker.createFromOptions(wasm, {
          baseOptions: { modelAssetPath: "/models/pose_landmarker_full.task", delegate: "CPU" },
          runningMode: "VIDEO",
          numPoses: 1,
        });
      }

      if (!video.duration || !Number.isFinite(video.duration)) {
        await new Promise<void>((resolve) =>
          video.addEventListener("loadedmetadata", () => resolve(), { once: true }),
        );
      }
      const duration = Math.min(video.duration, 18);
      const count = Math.max(20, Math.min(42, Math.round(duration * 3)));
      const samples: Array<{
        time: number;
        metrics: Record<MetricKey, number>;
        landmarks: Landmark[];
        releaseScore: number;
      }> = [];
      const side = hand === "right" ? [12, 14, 16, 24, 26, 28] : [11, 13, 15, 23, 25, 27];

      for (let index = 0; index < count; index += 1) {
        const time = 0.08 + ((Math.max(0.1, duration - 0.16) * index) / Math.max(1, count - 1));
        await seek(video, time);
        const result = landmarker.detectForVideo(video, index * 1000);
        const landmarks = result.landmarks?.[0] as Landmark[] | undefined;
        if (landmarks) {
          const [shoulder, elbow, wrist, hip, knee, ankle] = side.map((i) => landmarks[i]);
          const visibility = side.reduce((sum, i) => sum + (landmarks[i].visibility ?? 1), 0) / side.length;
          if (visibility > 0.35) {
            const metrics = {
              elbow: angle(shoulder, elbow, wrist),
              shoulder: angle(elbow, shoulder, hip),
              hip: angle(shoulder, hip, knee),
              knee: angle(hip, knee, ankle),
            };
            const releaseScore =
              metrics.elbow + Math.max(0, shoulder.y - wrist.y) * 160 + Math.max(0, hip.y - wrist.y) * 45;
            samples.push({ time, metrics, landmarks, releaseScore });
          }
        }
        setProgress(18 + Math.round(((index + 1) / count) * 72));
      }
      landmarker.close();
      if (samples.length < 3) throw new Error("NO_POSE");
      const release = samples.reduce((best, item) =>
        item.releaseScore > best.releaseScore ? item : best,
      );
      const target = references[reference];
      const averageDiff =
        metricKeys.reduce((sum, key) => sum + Math.abs(release.metrics[key] - target[key]), 0) /
        metricKeys.length;
      const score = Math.round(Math.max(42, Math.min(99, 100 - averageDiff * 1.75)));
      setProgress(100);
      setAnalysis({
        metrics: release.metrics,
        score,
        releaseTime: release.time,
        landmarks: release.landmarks,
        feedback: buildFeedback(release.metrics, target, lang),
      });
    } catch (cause) {
      setError(cause instanceof Error && cause.message === "NO_POSE" ? t.errorNoPose : t.errorGeneric);
    } finally {
      setLoading(false);
    }
  };

  const target = references[reference];

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Shooting Form Studio home">
          <span className="brand-ball" aria-hidden="true" />
          <span>FORM<span className="brand-slash">/</span>LAB</span>
        </a>
        <div className="top-actions">
          <span className="status"><i /> POSE MODEL READY</span>
          <button className="lang-toggle" onClick={() => setLang(lang === "ko" ? "en" : "ko")}>
            {lang === "ko" ? "EN" : "KR"}
          </button>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">{t.eyebrow}</p>
          <h1>
            <span>{t.heroA}</span>
            <strong>{t.heroB}</strong>
          </h1>
          <p className="hero-sub">{t.sub}</p>
          <div className="hero-actions">
            <a className="primary-link" href="#studio">{t.start}<span>↘</span></a>
            <a className="text-link" href="#method">{t.how}<span>↓</span></a>
          </div>
        </div>
        <div className="hero-visual" aria-label="Basketball release mechanics visualization">
          <div className="court-arc" />
          <div className="trajectory" />
          <div className="ball">●</div>
          <div className="pose-figure">
            <i className="head" />
            <i className="torso" />
            <i className="arm arm-a" />
            <i className="arm arm-b" />
            <i className="leg leg-a" />
            <i className="leg leg-b" />
            <b className="joint joint-1" />
            <b className="joint joint-2" />
            <b className="joint joint-3" />
            <b className="joint joint-4" />
            <b className="joint joint-5" />
          </div>
          <span className="angle-label angle-elbow">ELBOW<br/><b>176°</b></span>
          <span className="angle-label angle-knee">KNEE<br/><b>168°</b></span>
          <span className="frame-tag">RELEASE / 00:03.42</span>
        </div>
        <div className="privacy-note"><span>⌁</span>{t.private}</div>
      </section>

      <section className="studio" id="studio">
        <div className="section-index">01</div>
        <div className="studio-heading">
          <p>ANALYSIS STUDIO</p>
          <h2>{lang === "ko" ? "코트 밖의 슈팅 코치" : "Your coach, off the court"}</h2>
        </div>

        {!analysis ? (
          <div className="workbench">
            <div
              className={`upload-panel ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                hidden
                onChange={(e: ChangeEvent<HTMLInputElement>) => acceptFile(e.target.files?.[0])}
              />
              {videoUrl ? (
                <>
                  <video ref={videoRef} src={videoUrl} controls playsInline preload="metadata" />
                  <div className="file-meta">
                    <div><span className="file-dot" /><strong>{file?.name}</strong></div>
                    <button onClick={() => fileRef.current?.click()}>{t.replace}</button>
                  </div>
                </>
              ) : (
                <button className="drop-content" onClick={() => fileRef.current?.click()}>
                  <span className="upload-mark">↗</span>
                  <strong>{t.uploadTitle}</strong>
                  <small>{t.uploadSub}</small>
                  <em>{t.browse}</em>
                </button>
              )}
            </div>

            <aside className="settings-panel">
              <div className="panel-title"><span>02</span><h3>{t.settings}</h3></div>
              <fieldset>
                <legend>{t.hand}</legend>
                <div className="segmented">
                  {(["right", "left"] as Hand[]).map((value) => (
                    <button
                      key={value}
                      className={hand === value ? "active" : ""}
                      onClick={() => setHand(value)}
                    >
                      {value === "right" ? t.right : t.left}
                    </button>
                  ))}
                </div>
              </fieldset>
              <fieldset>
                <legend>{t.reference}</legend>
                <label className={`reference-option ${reference === "curry" ? "active" : ""}`}>
                  <input type="radio" checked={reference === "curry"} onChange={() => setReference("curry")} />
                  <span className="avatar">SC</span>
                  <span><strong>{t.curry}</strong><small>{t.modelNote}</small></span>
                  <b>↗</b>
                </label>
                <label className={`reference-option ${reference === "baseline" ? "active" : ""}`}>
                  <input type="radio" checked={reference === "baseline"} onChange={() => setReference("baseline")} />
                  <span className="avatar baseline-avatar">⌁</span>
                  <span><strong>{t.baseline}</strong><small>{t.baselineNote}</small></span>
                  <b>↗</b>
                </label>
              </fieldset>
              <button className="analyze-button" disabled={!file || loading} onClick={analyze}>
                {loading ? t.analyzing : t.analyze}
                <span>{loading ? `${progress}%` : "→"}</span>
              </button>
              {loading && (
                <div className="progress-wrap">
                  <div className="progress"><i style={{ width: `${progress}%` }} /></div>
                  <small>{t.modelLoading}</small>
                </div>
              )}
              {error && <p className="error-message">{error}</p>}
            </aside>
          </div>
        ) : (
          <div className="results">
            <div className="release-card">
              <div className="release-head">
                <span>{t.release} · {analysis.releaseTime.toFixed(2)}s</span>
                <span className="score-pill">{t.score} <b>{analysis.score}</b></span>
              </div>
              <canvas ref={canvasRef} />
              <video ref={videoRef} src={videoUrl} playsInline preload="auto" hidden />
            </div>
            <div className="metric-card">
              <div className="panel-title"><span>03</span><h3>{t.comparison}</h3></div>
              <div className="metric-head">
                <span />
                <span>{t.yours}</span>
                <span>{t.target}</span>
                <span>{t.delta}</span>
              </div>
              {metricKeys.map((key) => {
                const diff = analysis.metrics[key] - target[key];
                return (
                  <div className="metric-row" key={key}>
                    <strong>{t[key]}</strong>
                    <span>{analysis.metrics[key].toFixed(1)}°</span>
                    <span>{target[key].toFixed(1)}°</span>
                    <em className={Math.abs(diff) <= 7 ? "good" : ""}>
                      {diff > 0 ? "+" : ""}{diff.toFixed(1)}°
                    </em>
                  </div>
                );
              })}
            </div>
            <div className="coaching-card">
              <p>PRIORITY NOTES</p>
              <h3>{t.coaching}</h3>
              <ol>
                {analysis.feedback.map((item, index) => (
                  <li key={item}><span>0{index + 1}</span><p>{item}</p></li>
                ))}
              </ol>
              <button onClick={() => { setAnalysis(null); setError(""); }}>{t.again} <span>↗</span></button>
            </div>
          </div>
        )}
      </section>

      <section className="method" id="method">
        <div className="method-intro">
          <p>THE METHOD</p>
          <h2>{lang === "ko" ? "감이 아니라, 움직임을 봅니다." : "Not a hunch. Your actual movement."}</h2>
        </div>
        <article><span>{t.step1}</span><h3>{t.step1Title}</h3><p>{t.step1Body}</p><i>◇</i></article>
        <article><span>{t.step2}</span><h3>{t.step2Title}</h3><p>{t.step2Body}</p><i>⌖</i></article>
        <article><span>{t.step3}</span><h3>{t.step3Title}</h3><p>{t.step3Body}</p><i>↗</i></article>
      </section>

      <footer>
        <div className="brand"><span className="brand-ball" />FORM<span className="brand-slash">/</span>LAB</div>
        <p>{t.footer}</p>
        <a href="https://github.com/Rudwpahs/shooting-form-analysis" target="_blank" rel="noreferrer">
          SOURCE ↗
        </a>
      </footer>
    </main>
  );
}

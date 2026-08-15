"""Flask API + HTML front for angle-only shooting form analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .analyze import VIEW_TAGS, analyze_session, list_people_in_video, release_score
from .db import connect, ensure_seeded, list_player_angle_rows, list_player_catalog, save_session, upsert_player
from .similarity import match_player, match_views

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
ensure_seeded()

FEEDBACK_KO = {
    "elbow": "릴리스에서 팔꿈치를 조금 더 펴 에너지가 공까지 전달되게 하세요.",
    "shoulder": "슈팅 포켓을 살짝 높여 이마 앞에서 릴리스하세요.",
    "hip": "상체가 접히지 않게 골반·가슴을 림 방향으로 세워 주세요.",
    "knee": "릴리스까지 무릎 드라이브를 이어 위로 힘을 연결하세요.",
}


def _feedback(deltas: dict, lang: str = "ko") -> list:
    items = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out = []
    for key, diff in items:
        if abs(diff) <= 6:
            continue
        if lang == "ko":
            out.append(FEEDBACK_KO.get(key, f"{key} 각도를 조정해 보세요."))
        else:
            out.append(f"Adjust your {key} angle closer to the reference ({diff:+.1f}°).")
        if len(out) >= 3:
            break
    if not out:
        out.append("릴리스 각도가 기준과 잘 맞습니다. 같은 리듬을 반복하세요." if lang == "ko" else "Release angles look solid. Repeat the same rhythm.")
    return out


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/health")
def health():
    conn = connect()
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM player_angles").fetchone()["c"]
    finally:
        conn.close()
    return jsonify({"ok": True, "compare": "angles_deg_only", "players": n})


@app.get("/api/players")
def players():
    conn = connect()
    try:
        return jsonify({"players": list_player_catalog(conn)})
    finally:
        conn.close()


@app.post("/api/people")
def people():
    if "video" not in request.files and "video_0" not in request.files:
        return jsonify({"error": "Upload at least one video."}), 400
    file = request.files.get("video") or request.files.get("video_0")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        path = Path(tmp.name)
    try:
        previews = list_people_in_video(path)
        return jsonify(
            {
                "people": [
                    {
                        "person_index": p.person_index,
                        "bbox": p.bbox,
                        "thumb_jpeg_b64": p.thumb_jpeg_b64,
                    }
                    for p in previews
                ]
            }
        )
    finally:
        path.unlink(missing_ok=True)


def _collect_videos():
    videos = []
    for tag in VIEW_TAGS:
        key = f"video_{tag}"
        if key in request.files and request.files[key].filename:
            f = request.files[key]
            suffix = Path(f.filename or "clip.mp4").suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.save(tmp.name)
            tmp.close()
            videos.append((Path(tmp.name), tag))
    if videos:
        return videos

    files = []
    if "video" in request.files and request.files["video"].filename:
        files.append(request.files["video"])
    for i in range(3):
        key = f"video_{i}"
        if key in request.files and request.files[key].filename:
            files.append(request.files[key])
    views = request.form.getlist("views") or request.form.getlist("view")
    out = []
    for i, f in enumerate(files[:3]):
        tag = views[i] if i < len(views) and views[i] in VIEW_TAGS else (
            "side" if i == 0 else VIEW_TAGS[min(i, 2)]
        )
        suffix = Path(f.filename or "clip.mp4").suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.save(tmp.name)
        tmp.close()
        out.append((Path(tmp.name), tag))
    return out


def _query_views(session):
    views = [
        {"view": view.view, "space": view.space or "3d", "angles": view.release_angles}
        for view in session.views
        if view.release_angles and not view.error
    ]
    if not views and session.release_angles_merged:
        views = [{"view": "merged", "space": "3d", "angles": session.release_angles_merged}]
    return views


@app.post("/api/analyze")
def analyze():
    videos = _collect_videos()
    if not videos:
        return jsonify({"error": "Upload 1–3 videos."}), 400

    person_index = int(request.form.get("person_index", 0))
    auto_person = request.form.get("auto_person", "1") != "0"
    hand = request.form.get("hand") or None
    if hand not in ("left", "right", None):
        hand = None
    max_frames = min(400, max(60, int(request.form.get("max_frames", 240))))
    lang = request.form.get("lang") or "ko"
    target_player_key = (request.form.get("target_player_key") or "").strip()

    try:
        conn = connect()
        try:
            catalog = list_player_angle_rows(conn)
            available_keys = {str(row["player_key"]) for row in catalog}
            if target_player_key and target_player_key not in available_keys:
                return jsonify({"error": "Selected player does not have a compatible 3D profile."}), 400

            def run_one(index: int):
                sess = analyze_session(
                    videos,
                    person_index=index,
                    hand=hand,
                    max_frames=max_frames,
                    auto_person=False,
                )
                if sess.error and not sess.release_angles_merged:
                    return None
                if release_score(sess.release_angles_merged) <= 0:
                    return None
                qv = _query_views(sess)
                found = match_views(qv, catalog, top_k=5)
                if not found:
                    return None
                return sess, found

            packed = None
            if auto_person:
                best_key = None
                for index in range(3):
                    result = run_one(index)
                    if result is None:
                        continue
                    sess, found = result
                    key = (found[0].score, -found[0].distance_deg)
                    if best_key is None or key > best_key:
                        best_key = key
                        packed = result
            else:
                packed = run_one(person_index)

            if packed is None:
                session = analyze_session(
                    videos,
                    person_index=person_index,
                    hand=hand,
                    max_frames=max_frames,
                    auto_person=False,
                )
                if session.error and not session.release_angles_merged:
                    return jsonify({"error": session.error, "analysis": session.to_dict()}), 422
                query_views = _query_views(session)
                matches = match_views(query_views, catalog, top_k=5)
            else:
                session, matches = packed

            query_views = _query_views(session)
            match_dicts = [m.to_dict() for m in matches]
            top = matches[0] if matches else None
            selected = (
                match_player(query_views, catalog, target_player_key)
                if target_player_key
                else None
            )
            feedback_reference = selected or top
            feedback = _feedback(feedback_reference.deltas_deg, lang=lang) if feedback_reference else []
            session_id = save_session(
                conn,
                person_index=session.person_index,
                hand=hand or (session.views[0].hand if session.views else None),
                release_angles=session.release_angles_merged,
                views=[v.to_dict() for v in session.views],
            )
        finally:
            conn.close()

        return jsonify(
            {
                "session_id": session_id,
                "analysis": session.to_dict(),
                "matches": match_dicts,
                "closest_match": top.to_dict() if top else None,
                "selected_match": selected.to_dict() if selected else None,
                "target_player_key": target_player_key or None,
                "feedback": feedback,
                "note": "With 2–3 camera views, release angles come from multi-view triangulation (assumed front/side/oblique yaw). Single-view uses MediaPipe world landmarks. Similarity uses degrees only — not limb length or height.",
                "disclaimer": "Player angle profiles are unofficial self-measured estimates, not affiliated with any league or athlete.",
            }
        )
    finally:
        for path, _ in videos:
            path.unlink(missing_ok=True)


@app.post("/api/players")
def add_player():
    data = request.get_json(force=True, silent=True) or {}
    display_name = (data.get("display_name") or "").strip()
    angles = data.get("angles") or {}
    required = ("elbow", "shoulder", "hip", "knee")
    if not display_name or not all(k in angles for k in required):
        return jsonify({"error": "display_name and angles{elbow,shoulder,hip,knee} required"}), 400
    key = data.get("player_key") or display_name.lower().replace(" ", "_")
    conn = connect()
    try:
        upsert_player(
            conn,
            player_key=key,
            display_name=display_name,
            angles={k: float(angles[k]) for k in required},
            view=data.get("view") or "merged",
            hand=data.get("hand") or "right",
            source=data.get("source") or "self_measured",
            space=data.get("space") or "3d",
        )
    finally:
        conn.close()
    return jsonify({"ok": True, "player_key": key})


def main():
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

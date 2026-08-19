import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "models" / "youtube_single_view_pose_profiles.json"


def test_initial_roster_youtube_pose_profiles_are_complete_and_bounded():
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    assert payload["profile_count"] == 16
    assert payload["candidate_count"] == 16
    assert payload["rejected_count"] == 0

    for profile in payload["profiles"]:
        assert profile["model_status"] == "youtube_pose_candidate"
        assert profile["source_url"].startswith("https://www.youtube.com/watch?v=")
        assert profile["review"]["verdict"] == "ACCEPT"
        assert profile["review"]["player_match"] == "YES"
        assert profile["review"]["real_shot"] == "YES"
        assert profile["model_boundary"]["model_kind"] == "single_view_youtube_3d_pose_estimate"
        assert profile["model_boundary"]["calibration_status"] == "not_available"
        assert profile["model_boundary"]["metric_3d_status"] == "not_available"

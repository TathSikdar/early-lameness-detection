import streamlit as st
import os
import base64
import re
from models.architecture import AnomalyDetector
from utils.metrics import GaitData

st.set_page_config(layout="wide")

# ─────────────────────────────────────────────
# Global Variables
# ─────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "menu"

if "main_path" not in st.session_state:
    st.session_state.main_path = "data/processed"

def natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def build_all_cows(main_path: str):
    all_cows = []
    if os.path.exists(main_path):
        for session in sorted(os.listdir(main_path), key=natural_sort_key):
            session_path = os.path.join(main_path, session)
            if not os.path.isdir(session_path):
                continue
            for cow in sorted(os.listdir(session_path), key=natural_sort_key):
                cow_path = os.path.join(session_path, cow)
                if os.path.isdir(cow_path) and cow.startswith("cow_"):
                    all_cows.append((session, cow))
    return all_cows


st.session_state.all_cows = build_all_cows(st.session_state.main_path)

if "cow_index" not in st.session_state:
    st.session_state.cow_index = 0

st.session_state.total_cows = len(st.session_state.all_cows)

if st.session_state.total_cows > 0:
    st.session_state.cow_index = min(st.session_state.cow_index, st.session_state.total_cows - 1)
else:
    st.session_state.cow_index = 0

# Default lameness score of 10 means "not yet scored"
if "user_lameness_score" not in st.session_state:
    st.session_state.user_lameness_score = [10] * st.session_state.total_cows

# Load anomaly model
if "anomaly_model" not in st.session_state:
    model_path = "models/anomaly_model.pkl"
    if os.path.exists(model_path):
        detector = AnomalyDetector()
        detector.load_model(model_path)
        st.session_state.anomaly_model = detector
    else:
        st.session_state.anomaly_model = None


# ─────────────────────────────────────────────
# Page trigger: move to completion when all cows reviewed
# ─────────────────────────────────────────────

if st.session_state.cow_index == st.session_state.total_cows:
    st.session_state.page = "completion"


# ─────────────────────────────────────────────
# PAGE: Menu
# ─────────────────────────────────────────────

if st.session_state.page == "menu":
    st.title("🐄 Cow Lameness Detection Dashboard")
    st.markdown(f"**Total cows to review:** {st.session_state.total_cows}")

    if st.button("Start Review"):
        st.session_state.page = "review"
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: Review
# ─────────────────────────────────────────────

elif st.session_state.page == "review":

    import json
    import datetime as dt
    session, cow_folder = st.session_state.all_cows[st.session_state.cow_index]
    current_cow_path = os.path.join(st.session_state.main_path, session, cow_folder)
    lameness_json_path = os.path.join(current_cow_path, "lameness_analysis.json")
    cow_data = None
    if os.path.exists(lameness_json_path):
        with open(lameness_json_path, "r") as f:
            cow_data = json.load(f)
    else:
        st.warning(f"No lameness_analysis.json found for {cow_folder}")
        cow_data = {}

    # ── Header ──
    st.title("🐄 Cow Lameness Review")
    st.markdown(f"**Session:** `{session}` &nbsp;|&nbsp; **Cow:** `{cow_folder}` &nbsp;|&nbsp; **Progress:** {st.session_state.cow_index + 1} / {st.session_state.total_cows}")
    st.progress((st.session_state.cow_index) / st.session_state.total_cows)
    st.divider()

    # ── Videos: top / side / front ──
    st.subheader("📹 Cow Gait Videos")
    video_angles = ["top", "side", "front"]
    col_top, col_side, col_front = st.columns(3)
    cols = {"top": col_top, "side": col_side, "front": col_front}

    def render_video(video_path, label):
        ext = os.path.splitext(video_path)[-1].lower().replace(".", "")
        mime = "video/mp4" if ext == "mp4" else "video/quicktime"
        st.markdown(f"**{label}**")
        with open(video_path, "rb") as f:
            st.video(f.read(), format=mime, autoplay=True, loop=True, muted=True)

    for angle in video_angles:
        angle_path = os.path.join(current_cow_path, angle)
        with cols[angle]:
            if os.path.exists(angle_path):
                video_files = sorted([
                    f for f in os.listdir(angle_path)
                    if f.lower().endswith((".mp4", ".mov"))
                ], key=natural_sort_key)
                if video_files:
                    video_path = os.path.join(angle_path, video_files[0])
                    render_video(video_path, f"📷 {angle.capitalize()} View")
                else:
                    st.markdown(f"**{angle.capitalize()} View**")
                    st.info("No video found")
            else:
                st.markdown(f"**{angle.capitalize()} View**")
                st.warning(f"Folder not found: `{angle_path}`")
    st.divider()


    # ── Model & JSON Scores ──
    st.subheader("🤖 Model & JSON Predictions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Lameness Score", cow_data.get("lameness_score", "-"))
    col2.metric("Head Bob Score", f"{round(cow_data.get('head_bob_score', 0), 2) if cow_data.get('head_bob_score') is not None else '-'}")
    col3.metric("Spine Score", f"{round(cow_data.get('spine_score', 0), 2) if cow_data.get('spine_score') is not None else '-'}")
    st.markdown(f"**Needs Review:** `{cow_data.get('needs_review', '-')}` | **Review Priority:** `{cow_data.get('review_priority', '-')}`")
    st.markdown(f"**Notes:** {cow_data.get('notes', '')}")
    st.divider()

    # ── Correction Inputs ──
    st.subheader("🧑‍⚕️ Correction Inputs")
    corrected_lameness = st.number_input(
        "Corrected Lameness Score (0-5)", min_value=0, max_value=5,
        value=cow_data.get("corrected_lameness_score") if cow_data.get("corrected_lameness_score") is not None else cow_data.get("lameness_score", 0),
        key=f"corrected_lameness_{session}_{cow_folder}"
    )
    # Removed duplicate corrected ear tag field
    notes = st.text_area(
        "Correction Notes", value=cow_data.get("notes") or "",
        key=f"notes_{session}_{cow_folder}"
    )
    st.divider()



    # ── Ear Tag Images Collage (from ear_tag folder) ──
    st.subheader("🏷️ Cow Ear Tag Frames")
    ear_tag_folder = os.path.join(current_cow_path, "ear_tag")
    if os.path.exists(ear_tag_folder):
        ear_tag_imgs = sorted([
            f for f in os.listdir(ear_tag_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ], key=natural_sort_key)
        if ear_tag_imgs:
            cols = st.columns(3)
            for i, img_file in enumerate(ear_tag_imgs):
                img_path = os.path.join(ear_tag_folder, img_file)
                with cols[i % 3]:
                    st.image(img_path, width=300)
        else:
            st.info("No ear tag images found in ear_tag folder")
    else:
        st.info("No ear_tag folder found for this cow")

    # ── Cow RFID (single correction field) ──
    cow_id = "ASDF-123"  # TODO: replace with CV model output
    corrected_cow_id = st.text_input(
        "Corrected Cow ID (Ear Tag)",
        value=cow_data.get("corrected_cow_id", ""),
        key=f"corrected_cow_id_{session}_{cow_folder}"
    )
    st.markdown(f"**Cow RFID:** `{cow_id}`")

    st.divider()


    # ── Submit ──
    if st.button("✅ Submit & Next Cow"):
        # Update lameness_analysis.json with corrections
        cow_data["corrected_lameness_score"] = int(corrected_lameness)
        cow_data["correction_timestamp"] = dt.datetime.now().isoformat()
        cow_data["corrected_cow_id"] = corrected_cow_id
        cow_data["notes"] = notes
        with open(lameness_json_path, "w") as f:
            json.dump(cow_data, f, indent=2)
        st.session_state.cow_index += 1
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: Completion
# ─────────────────────────────────────────────

elif st.session_state.page == "completion":

    st.title("✅ Review Complete!")
    st.markdown(f"All **{st.session_state.total_cows}** cows have been reviewed.")

    # Write scores to file
    with open("user_lameness_score.txt", "w") as f:
        for i, (session, cow) in enumerate(st.session_state.all_cows):
            score = st.session_state.user_lameness_score[i]
            f.write(f"{session},{cow},{score}\n")

    st.success("Scores saved to `user_lameness_score.txt`")

    # Show summary table
    st.subheader("📋 Score Summary")
    summary_data = {
        "Session": [s for s, _ in st.session_state.all_cows],
        "Cow": [c for _, c in st.session_state.all_cows],
        "Your Score": st.session_state.user_lameness_score,
    }
    st.dataframe(summary_data, use_container_width=True)

    if st.button("🔄 Start Over"):
        for key in ["page", "cow_index", "all_cows", "total_cows", "user_lameness_score"]:
            del st.session_state[key]
        st.rerun()
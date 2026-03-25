import streamlit as st
import os
import base64
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

# Build a flat list of (session, cow) tuples from all sessions
if "all_cows" not in st.session_state:
    all_cows = []
    main_path = st.session_state.main_path
    if os.path.exists(main_path):
        for session in sorted(os.listdir(main_path)):
            session_path = os.path.join(main_path, session)
            if os.path.isdir(session_path):
                for cow in sorted(os.listdir(session_path)):
                    cow_path = os.path.join(session_path, cow)
                    if os.path.isdir(cow_path):
                        all_cows.append((session, cow))
    st.session_state.all_cows = all_cows

if "cow_index" not in st.session_state:
    st.session_state.cow_index = 0

if "total_cows" not in st.session_state:
    st.session_state.total_cows = len(st.session_state.all_cows)

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

    session, cow = st.session_state.all_cows[st.session_state.cow_index]
    current_cow_path = os.path.join(st.session_state.main_path, session, cow)

    # ── Header ──
    st.title("🐄 Cow Lameness Review")
    st.markdown(f"**Session:** `{session}` &nbsp;|&nbsp; **Cow:** `{cow}` &nbsp;|&nbsp; **Progress:** {st.session_state.cow_index + 1} / {st.session_state.total_cows}")
    st.progress((st.session_state.cow_index) / st.session_state.total_cows)
    st.divider()

    # ── Videos: top / side / front ──
    st.subheader("📹 Cow Gait Videos")
    video_angles = ["top", "side", "front"]
    col_top, col_side, col_front = st.columns(3)
    cols = {"top": col_top, "side": col_side, "front": col_front}

    def autoplay_video(video_path, label):
        """Render an autoplaying, looping, muted video using base64 HTML."""
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode()
        ext = os.path.splitext(video_path)[-1].lower().replace(".", "")
        mime = "video/mp4" if ext == "mp4" else "video/quicktime"
        html = f"""
        <p style="margin-bottom:6px; font-weight:600; font-size:16px">{label}</p>
        <video width="100%" autoplay loop muted playsinline
               style="border-radius:10px; background:#000; object-fit:contain;">
            <source src="data:{mime};base64,{b64}" type="{mime}">
        </video>
        """
        st.html(html)

    for angle in video_angles:
        angle_path = os.path.join(current_cow_path, angle)
        with cols[angle]:
            if os.path.exists(angle_path):
                video_files = [
                    f for f in os.listdir(angle_path)
                    if f.lower().endswith((".mp4", ".mov"))
                ]
                if video_files:
                    video_path = os.path.join(angle_path, video_files[0])
                    autoplay_video(video_path, f"📷 {angle.capitalize()} View")
                else:
                    st.markdown(f"**{angle.capitalize()} View**")
                    st.info("No video found")
            else:
                st.markdown(f"**{angle.capitalize()} View**")
                st.warning(f"Folder not found: `{angle_path}`")

    st.divider()

    # ── Model Scores ──
    st.subheader("🤖 Model Predictions")
    
    predicted_score = 0
    confidence = 0
    if st.session_state.anomaly_model:
        keypoints_csv = os.path.join(current_cow_path, "keypoints.csv")
        if os.path.exists(keypoints_csv):
            try:
                gait = GaitData(keypoints_csv)
                features = gait.extract_features()
                anomaly_score = st.session_state.anomaly_model.predict(features)
                # Map anomaly score (0-1) to lameness (0-5)
                predicted_score = int(anomaly_score * 5)
                confidence = 1 - anomaly_score  # Higher confidence for normal
            except Exception as e:
                st.warning(f"Error computing prediction: {e}")
        else:
            st.info("No keypoints available for prediction")
    else:
        st.info("Anomaly model not trained yet")

    col_lame, col_conf = st.columns(2)
    col_lame.metric("Model Lameness Score", predicted_score)
    col_conf.metric("Model Confidence Score", f"{confidence:.2f}")

    st.divider()

    # ── User Lameness Score Input ──
    st.subheader("🧑‍⚕️ Your Lameness Score")

    current_score = st.session_state.user_lameness_score[st.session_state.cow_index]
    if current_score != 10:
        st.success(f"Current selection: **{current_score}**")
    else:
        st.info("No score selected yet — pick one below")

    btn_cols = st.columns(6)
    for score_val, col in enumerate(btn_cols):
        if col.button(str(score_val), key=f"score_{score_val}"):
            st.session_state.user_lameness_score[st.session_state.cow_index] = score_val
            st.rerun()

    st.divider()

    # ── Ear Tag Images ──
    st.subheader("🏷️ Cow Ear Tag Images")
    image_folder_path = os.path.join(current_cow_path, "images")

    if os.path.exists(image_folder_path):
        image_files = [
            f for f in os.listdir(image_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if image_files:
            col_eartags = st.columns(3)
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_folder_path, image_file)
                col_eartags[i % 3].image(image_path)
        else:
            st.info("No ear tag images found")
    else:
        st.info("No images folder found for this cow")

    # ── Cow RFID ──
    cow_id = "ASDF-123"  # TODO: replace with CV model output
    st.markdown(f"**Cow RFID:** `{cow_id}`")

    st.divider()

    # ── Submit ──
    if st.session_state.user_lameness_score[st.session_state.cow_index] == 10:
        st.warning("⚠️ Please select a lameness score (0–5) before submitting.")

    if st.button("✅ Submit & Next Cow", disabled=(st.session_state.user_lameness_score[st.session_state.cow_index] == 10)):
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
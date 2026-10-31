import tempfile
import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from backend import ExerciseAnalyzer, process_frame

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='GymvisionAI Enterprise | Biomechanical Engine',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# --- WORLD-CLASS DESIGN SYSTEM (Obsidian & Neon Mint) ---
st.markdown(
    """
    <style>
    /* Global Theme */
    .stApp {
        background-color: #07090b;
        color: #e5e7eb;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean Enterprise Header Card */
    .header-container {
        background: linear-gradient(135deg, #111822 0%, #0b0f15 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #00f59b;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: #9ca3af;
        margin: 4px 0 0 0;
    }

    .system-badge {
        display: inline-block;
        background: rgba(0, 245, 155, 0.1);
        border: 1px solid rgba(0, 245, 155, 0.3);
        color: #00f59b;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
        white-space: nowrap;
    }

    /* High-End Glass Telemetry Cards */
    .metric-card {
        background: rgba(17, 24, 32, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 2px solid #00f59b;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    
    .metric-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #9ca3af;
        font-weight: 700;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #00f59b;
        margin-top: 4px;
        text-shadow: 0 0 15px rgba(0, 245, 155, 0.2);
    }
    
    /* Professional Status Boxes */
    .status-box-good {
        background: rgba(0, 245, 155, 0.06);
        border: 1px solid rgba(0, 245, 155, 0.2);
        border-left: 4px solid #00f59b;
        padding: 14px;
        border-radius: 8px;
        color: #00f59b;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .status-box-warning {
        background: rgba(248, 113, 113, 0.06);
        border: 1px solid rgba(248, 113, 113, 0.2);
        border-left: 4px solid #f87171;
        padding: 14px;
        border-radius: 8px;
        color: #fca5a5;
        font-weight: 700;
        font-size: 1rem;
    }

    /* Sidebar Clean-up */
    section[data-testid="stSidebar"] {
        background-color: #0b0f15;
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    h1, h2, h3 {
        color: #ffffff;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- LOAD YOLO MODEL (Cached) ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolo11n-pose.pt')


model = load_yolo_model()

# --- SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.image('https://img.icons8.com/neon/96/dumbbell.png', width=60)
    st.markdown('### **GymvisionAI OS**')
    st.markdown(
        '<span style="color: #9ca3af; font-size: 0.8rem;">Enterprise Biometric Suite v2.4</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    app_mode = st.radio(
        '🔌 Input Source', ['Live Webcam Stream', 'Upload Video File']
    )

    st.divider()

    exercise_choice = st.selectbox(
        '🎯 Target Exercise',
        options=['squat', 'pushup'],
        format_func=lambda x: x.capitalize(),
    )

    st.divider()

    st.markdown('### ⚙️ Parameters')
    confidence_threshold = st.slider(
        'Confidence Threshold', 0.1, 0.9, 0.5, 0.05
    )

    if 'analyzer' not in st.session_state or st.session_state.analyzer.exercise_type != exercise_choice:
        st.session_state.analyzer = ExerciseAnalyzer(exercise_type=exercise_choice)

    reset_btn = st.button('🔄 Reset Active Set', use_container_width=True)
    if reset_btn:
        st.session_state.analyzer.start_set()
        st.toast('Session telemetry recalibrated.', icon='⚡')

    st.divider()
    st.markdown(
        '<p style="font-size: 0.7rem; color: #4b5563; text-align: center;">Powered by YOLO11-Pose<br>© 2026 GymvisionAI Corp.</p>',
        unsafe_allow_html=True,
    )

# --- REBUILT HEADER (Using Streamlit Columns for Perfect Alignment) ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown(
        """
        <div class="header-container" style="margin-bottom: 0px; border-left: 4px solid #00f59b;">
            <h1 class="header-title">Biomechanical Form & Telemetry Suite</h1>
            <p class="header-subtitle">Real-time computer vision skeleton tracking & kinematic angle feedback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        """
        <div class="header-container" style="margin-bottom: 0px; text-align: center; display: flex; align-items: center; justify-content: center; height: 100%;">
            <div>
                <span class="system-badge">● System Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write('') # Spacing

# --- MAIN DASHBOARD LAYOUT ---
col_video, col_telemetry = st.columns([2.4, 1])

with col_telemetry:
    st.markdown('### 📊 Live Telemetry')

    rep_placeholder = st.empty()
    stage_placeholder = st.empty()
    duration_placeholder = st.empty()

    st.markdown('### 🧠 Kinematic Feedback')
    form_status_placeholder = st.empty()

    st.markdown('---')
    st.info(
        '💡 **Coach Insight:** Maintain rigid core tension and a controlled 2-second eccentric descent.'
    )

with col_video:
    if app_mode == 'Live Webcam Stream':
        st.markdown('### 🎥 Live Video Stream & Skeleton Tracking')
        video_container = st.empty()
        start_camera = st.checkbox(
            '🔴 Initialize Camera Stream & Begin Workout', value=False
        )
    else:
        st.markdown('### 📁 Video File Processing Pipeline')
        uploaded_file = st.file_uploader(
            'Upload session recording (MP4, AVI, MOV)', type=['mp4', 'avi', 'mov']
        )
        video_container = st.empty()
        process_video_btn = st.button(
            '🚀 Run Frame-Accurate Analysis', use_container_width=True
        )


# --- LOGIC 1: WEBCAM STREAM ---
if app_mode == 'Live Webcam Stream' and locals().get('start_camera', False):
    cap = cv2.VideoCapture(0)
    analyzer = st.session_state.analyzer
    analyzer.fps = 30

    if not analyzer.is_active:
        analyzer.start_set()

    try:
        while start_camera and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error('Unable to access webcam feed.')
                break

            frame = cv2.flip(frame, 1)
            annotated_frame = process_frame(model, frame, analyzer)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_container.image(
                rgb_frame, channels='RGB', use_container_width=True
            )

            rep_placeholder.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Completed Repetitions</div>
                    <div class="metric-value">{analyzer.counter}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            stage_placeholder.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Movement Phase</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: #00f59b;">{analyzer.stage.upper() if analyzer.stage else "READY"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            duration_placeholder.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Set Active Duration</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: #38bdf8;">{analyzer.set_duration}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if (
                analyzer.form_status == 'Correct Form'
                or analyzer.form_status == 'Ready'
            ):
                form_status_placeholder.markdown(
                    f'<div class="status-box-good">✅ {analyzer.form_status}</div>',
                    unsafe_allow_html=True,
                )
            else:
                form_status_placeholder.markdown(
                    f'<div class="status-box-warning">⚠️ {analyzer.form_status}</div>',
                    unsafe_allow_html=True,
                )

            time.sleep(0.02)

    except Exception as e:
        st.error(f'Stream exception: {e}')
    finally:
        cap.release()


# --- LOGIC 2: VIDEO FILE UPLOAD PROCESSING ---
elif app_mode == 'Upload Video File' and uploaded_file is not None:
    if process_video_btn:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        if file_fps <= 0 or np.isnan(file_fps):
            file_fps = 30

        analyzer = st.session_state.analyzer
        analyzer.fps = file_fps
        analyzer.start_set()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = process_frame(model, frame, analyzer)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                video_container.image(
                    rgb_frame, channels='RGB', use_container_width=True
                )

                rep_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Completed Repetitions</div>
                        <div class="metric-value">{analyzer.counter}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                stage_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Movement Phase</div>
                        <div class="metric-value" style="font-size: 1.4rem; color: #00f59b;">{analyzer.stage.upper() if analyzer.stage else "READY"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                duration_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Set Active Duration</div>
                        <div class="metric-value" style="font-size: 1.4rem; color: #38bdf8;">{analyzer.set_duration}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if (
                    analyzer.form_status == 'Correct Form'
                    or analyzer.form_status == 'Ready'
                ):
                    form_status_placeholder.markdown(
                        f'<div class="status-box-good">✅ {analyzer.form_status}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    form_status_placeholder.markdown(
                        f'<div class="status-box-warning">⚠️ {analyzer.form_status}</div>',
                        unsafe_allow_html=True,
                    )

            st.success(
                f'🎉 Analysis Complete — Certified Reps: **{analyzer.counter}** | Duration: **{analyzer.set_duration}s**'
            )

        except Exception as e:
            st.error(f'Video processing error: {e}')
        finally:
            cap.release()
    else:
        video_container.info(
            '📁 File loaded. Click **Run Frame-Accurate Analysis** in the sidebar to start processing.'
        )

else:
    # Standby UI State
    if app_mode == 'Live Webcam Stream':
        video_container.info(
            '🔴 Check **Initialize Camera Stream & Begin Workout** in the sidebar to start feed.'
        )
    else:
        video_container.info(
            '📁 Upload an MP4, MOV, or AVI video file using the sidebar to begin analysis.'
        )

    rep_placeholder.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Completed Repetitions</div>
            <div class="metric-value">0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage_placeholder.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Movement Phase</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #6b7280;">STANDBY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    duration_placeholder.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Set Active Duration</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #6b7280;">0s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_status_placeholder.markdown(
        '<div class="status-box-good" style="background: rgba(17, 24, 32, 0.7); border-left-color: #4b5563; color: #9ca3af;">System Standby — Awaiting Input Signal</div>',
        unsafe_allow_html=True,
    )
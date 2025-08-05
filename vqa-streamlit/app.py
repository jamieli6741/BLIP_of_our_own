import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import time

# set page configuration
st.set_page_config(
    page_title="BLIP Visual Question Answering",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# define CSS styles
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("BLIP Visual Question Answering System")
st.markdown("Upload an image and ask me any question about it!")

# 初始化session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.processor = None
    st.session_state.model = None
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""


@st.cache_resource
def load_model():
    """Load BLIP model (cached to avoid reloading)"""
    with st.spinner("Loading BLIP model, this may take a few minutes on first run..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        model.eval()
    return processor, model, device


# load model
if not st.session_state.model_loaded:
    try:
        processor, model, device = load_model()
        st.session_state.processor = processor
        st.session_state.model = model
        st.session_state.device = device
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

# create two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Supports JPG, JPEG, PNG, BMP, WEBP formats"
    )

    if uploaded_file is not None:
        # show image uploaded - 修复警告：使用 use_container_width 替代 use_column_width
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded image", use_container_width=True)

        # save image to session state
        st.session_state.current_image = image

with col2:
    st.subheader("❓ Ask Questions")

    # initialize session state to store the selected question
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""

    # default questions
    st.markdown("**Quick Questions:**")
    col_q1, col_q2 = st.columns(2)

    with col_q1:
        if st.button("What's in the image?"):
            st.session_state.selected_question = "What is in the image?"
        if st.button("How many people?"):
            st.session_state.selected_question = "How many people are there?"
        if st.button("What color is it?"):
            st.session_state.selected_question = "What color is this?"

    with col_q2:
        if st.button("What are they doing?"):
            st.session_state.selected_question = "What are they doing?"
        if st.button("Where is this?"):
            st.session_state.selected_question = "Where is this?"
        if st.button("What time is it?"):
            st.session_state.selected_question = "What time is it?"

    # get question input
    custom_question = st.text_input(
        "Enter your question",
        placeholder="e.g., How many people are in the image?",
        help="You can ask any question about the image content"
    )

    if st.session_state.selected_question and not custom_question:
        question = st.session_state.selected_question
        st.info(f"Using preset question: {question}")
    elif custom_question:
        question = custom_question
        st.session_state.selected_question = ""  # remove default questions
    else:
        question = ""

st.markdown("---")

# question button and answer present
if st.button("🔍 Get Answer", type="primary", disabled=not uploaded_file or not question):
    if uploaded_file and question:
        with st.spinner("Thinking..."):
            try:
                # use blip
                inputs = st.session_state.processor(
                    st.session_state.current_image,
                    question,
                    return_tensors="pt"
                ).to(st.session_state.device)

                # generate answer
                start_time = time.time()
                with torch.no_grad():
                    outputs = st.session_state.model.generate(**inputs, max_length=50)

                # decode answer
                answer = st.session_state.processor.decode(outputs[0], skip_special_tokens=True)
                processing_time = time.time() - start_time

                # show results
                st.markdown("### Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # additional info
                with st.expander("Details"):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
                    st.write(f"**Processing time:** {processing_time:.2f} seconds")
                    st.write(f"**Device:** {st.session_state.device}")

            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
    else:
        st.warning("Please upload an image and enter a question")

# sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is a Visual Question Answering system based on BLIP (Bootstrapping Language-Image Pre-training) model.

    **Features:**
    - Supports multiple image formats
    - Uses pre-trained BLIP model
    - GPU acceleration support
    - Simple and easy-to-use interface

    **Model Information:**
    - Model: Salesforce/blip-vqa-base
    - Task: Visual Question Answering
    - Language: English
    """)

    st.markdown("---")

    st.header("Tips")
    st.markdown("""
    1. Upload clear images for better results
    2. Keep questions concise and clear
    3. You can ask about objects, colors, counts, actions, etc.
    4. Model answers in English
    """)

    # 显示系统信息
    st.markdown("---")
    st.header("⚙System Info")
    if st.session_state.model_loaded:
        st.info(f"Device: {st.session_state.device}")
        st.info(f"✅ Model Status: Loaded")
    else:
        st.info(f"❌ Model Status: Not loaded")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>Built with BLIP model | Powered by Streamlit</div>",
    unsafe_allow_html=True
)
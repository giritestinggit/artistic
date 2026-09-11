"""
Artistic Image Style Transfer - Streamlit Application
Universal entry point for local execution and Streamlit Community Cloud.
"""

import os
import sys
import time
import glob
import io
from PIL import Image

import streamlit as st

# 1. Page Configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Artistic Image Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# Base path setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torchvision.transforms as transforms

# Limit CPU threads to avoid pegging resources
torch.set_num_threads(2)

from paths import (
    ensure_weights_exist,
    SAMPLES_DIR,
    COCO_DIR,
    WIKIART_CACHE_DIR,
    OUTPUTS_DIR
)
from inference.stylize import get_model
from models.adain import adaptive_instance_normalization
from data.preprocessing import tensor_to_image, rgb_to_ycbcr, ycbcr_to_rgb
from evaluation.metrics import evaluate_transfer_pair


@st.cache_resource(show_spinner=False)
def load_cached_model():
    """Ensures checkpoints exist, loads model weights, and caches model in memory."""
    ensure_weights_exist()
    model = get_model(force_reload=False)
    with torch.no_grad():
        test_in = torch.full((1, 3, 64, 64), 0.5)
        test_out = model(test_in, test_in, 0.5)
        if test_out.mean().item() < 0.05:
            from models.pretrained import load_pretrained_decoder
            from paths import DECODER_WEIGHTS_PATH
            model.decoder = load_pretrained_decoder(DECODER_WEIGHTS_PATH)
    return model


def main():

    st.title("🎨 Artistic Image Style Transfer")
    st.caption("Fast neural style transfer with controllable strength and strict content preservation.")

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")

    if st.sidebar.button("🔄 Clear Cache & Reload Weights"):
        st.cache_resource.clear()
        st.rerun()

    strength = st.sidebar.slider(
        "Style Strength (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="0.0 = pure original content, 1.0 = maximum artistic brushstrokes."
    )

    color_preserve = st.sidebar.checkbox(
        "Preserve Content Colors",
        value=False,
        help="Applies artistic texture while strictly preserving original content colors (YCbCr mode)."
    )

    res_choice = st.sidebar.select_slider(
        "Processing Resolution",
        options=[256, 384, 512],
        value=256,
        format_func=lambda x: f"{x} × {x} (Fast ~3–5s)" if x == 256 else (f"{x} × {x} (Balanced ~8s)" if x == 384 else f"{x} × {x} (High Res ~20s)"),
        help="256×256 is recommended for real-time responsiveness on CPU."
    )

    # Two main tabs: 1. Style Transfer & Embed, 2. Decode Secret Message
    tab_stylize, tab_decode = st.tabs(["🎨 Style Transfer & Hide Message", "🕵️ Decode Hidden Message"])

    with tab_stylize:
        # Two-column layout for input selection
        col1, col2 = st.columns(2)

        # Content Image Input
        with col1:
            st.subheader("1. Content Image")
            content_source = st.radio("Content Source", ["Sample Library", "Upload Image"], horizontal=True)
            content_img = None

            if content_source == "Sample Library":
                sample_files = []
                if os.path.exists(COCO_DIR):
                    sample_files = sorted(glob.glob(os.path.join(COCO_DIR, "*.jpg")))[:30]
                if not sample_files and os.path.exists(os.path.join(SAMPLES_DIR, "content")):
                    sample_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "content", "*.jpg")))

                if sample_files:
                    selected_sample = st.selectbox("Choose Content Sample", sample_files, format_func=os.path.basename)
                    content_img = Image.open(selected_sample).convert("RGB")
                    st.image(content_img, caption="Content Image", use_container_width=True)
                else:
                    st.info("No sample files found. Please upload an image.")
            else:
                uploaded_c = st.file_uploader("Upload Content Photo", type=["jpg", "jpeg", "png"])
                if uploaded_c:
                    content_img = Image.open(uploaded_c).convert("RGB")
                    st.image(content_img, caption="Uploaded Content", use_container_width=True)

        # Style Image Input
        with col2:
            st.subheader("2. Artistic Style")
            style_source = st.radio("Style Source", ["Artistic Library", "Upload Image"], horizontal=True)
            style_img = None

            if style_source == "Artistic Library":
                # Search in wikiart_cache first (if local), fallback to samples/styles (cloud & local)
                source_dir = WIKIART_CACHE_DIR if (os.path.exists(WIKIART_CACHE_DIR) and os.listdir(WIKIART_CACHE_DIR)) else os.path.join(SAMPLES_DIR, "styles")
                categories = []
                if os.path.exists(source_dir):
                    categories = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

                if categories:
                    def format_genre(g):
                        names = {
                            "Cubism": "📐 Cubism",
                            "Expressionism": "🎭 Expressionism",
                            "Impressionism": "🎨 Impressionism",
                            "Post_Impressionism": "🌌 Post-Impressionism",
                            "Ukiyo_e": "🌊 Ukiyo-e (Woodblock)"
                        }
                        return names.get(g, g.replace("_", " ").title())

                    chosen_cat = st.selectbox("Art Movement / Genre", categories, format_func=format_genre)
                    cat_files = sorted(glob.glob(os.path.join(source_dir, chosen_cat, "*.*")))
                    if cat_files:
                        def format_art(f):
                            base = os.path.splitext(os.path.basename(f))[0]
                            if "_" in base:
                                artist, title = base.split("_", 1)
                                return f"{artist.replace('-', ' ').title()} — {title.replace('-', ' ').title()}"
                            return base.replace('-', ' ').title()

                        chosen_style_file = st.selectbox("Artwork", cat_files, format_func=format_art)
                        style_img = Image.open(chosen_style_file).convert("RGB")
                        st.image(style_img, caption=f"Style: {format_genre(chosen_cat)}", use_container_width=True)
                else:
                    fallback_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "styles", "**", "*.*"), recursive=True))
                    if fallback_files:
                        chosen_style_file = st.selectbox("Artwork", fallback_files, format_func=os.path.basename)
                        style_img = Image.open(chosen_style_file).convert("RGB")
                        st.image(style_img, caption="Artwork Sample", use_container_width=True)
                    else:
                        st.info("No style samples found. Please upload a style image.")
            else:
                uploaded_s = st.file_uploader("Upload Style Artwork", type=["jpg", "jpeg", "png"])
                if uploaded_s:
                    style_img = Image.open(uploaded_s).convert("RGB")
                    st.image(style_img, caption="Uploaded Artwork", use_container_width=True)

        st.markdown("---")

        # Steganography Message Embedding Input (Optional Add-on)
        with st.expander("🔒 Hide Secret Message in Artwork (Steganography)", expanded=False):
            st.caption("Invisibly embed a secret message into your stylized artwork using LSB encoding. The recipient can decode it in the 'Decode' tab.")
            stego_message = st.text_area(
                "Secret Message to Hide",
                placeholder="e.g. i want to meet u",
                help="This message will be invisibly encoded into the pixel values of your stylized artwork."
            )
            stego_password = st.text_input(
                "Secret Passcode / Key (Optional)",
                type="password",
                placeholder="Leave blank for unencrypted message",
                help="If set, only someone who enters this exact passcode can read the message."
            )
            if stego_message.strip():
                st.info(f"📝 Message: **{len(stego_message.strip())} characters** ({len(stego_message.strip().encode('utf-8'))} bytes)")

        # Session state initialization for fast slider caching
        if "cached_content_id" not in st.session_state:
            st.session_state.cached_content_id = None
            st.session_state.cached_style_id = None
            st.session_state.last_result = None
            st.session_state.last_time = 0.0
            st.session_state.has_stego = False
            st.session_state.stego_len = 0
            st.session_state.stego_encrypted = False

        # Stylization Action Button
        if st.button("🚀 Apply Artistic Style Transfer", type="primary", use_container_width=True):
            if content_img is None:
                st.error("Please select or upload a Content Image.")
            elif style_img is None:
                st.error("Please select or upload a Style Image.")
            else:
                t_start = time.time()
                with st.spinner(f"Loading model & stylizing image at {res_choice}×{res_choice}..."):
                    model = load_cached_model()
                    to_tensor = transforms.ToTensor()
                    device = torch.device("cpu")

                    c_resized = content_img.resize((res_choice, res_choice))
                    s_resized = style_img.resize((res_choice, res_choice))
                    c_t = to_tensor(c_resized).unsqueeze(0).to(device)
                    s_t = to_tensor(s_resized).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        if color_preserve:
                            out_tensor = model.stylize_color_preserving(c_t, s_t, alpha=strength)
                        else:
                            out_tensor = model(c_t, s_t, alpha=strength)

                    result_img = tensor_to_image(out_tensor)

                    # Embed secret message if provided
                    if stego_message.strip():
                        from steganography import embed_message
                        pwd = stego_password.strip() if stego_password.strip() else None
                        try:
                            result_img = embed_message(result_img, stego_message.strip(), password=pwd)
                            st.session_state.has_stego = True
                            st.session_state.stego_len = len(stego_message.strip())
                            st.session_state.stego_encrypted = bool(pwd)
                        except Exception as e:
                            st.warning(f"Could not embed secret message: {e}")
                            st.session_state.has_stego = False
                    else:
                        st.session_state.has_stego = False

                t_elapsed = round(time.time() - t_start, 2)
                st.session_state.last_result = result_img
                st.session_state.last_time = t_elapsed
                st.session_state.last_content = content_img
                st.session_state.last_style = style_img

        # Display Results
        if st.session_state.last_result is not None:
            st.success(f"⚡ Stylization complete in **{st.session_state.last_time} seconds**!")

            if st.session_state.get("has_stego", False):
                enc_tag = " (Passcode Protected 🔑)" if st.session_state.get("stego_encrypted", False) else ""
                st.info(f"🔒 **Steganography Active**: Hidden message ({st.session_state.get('stego_len', 0)} chars){enc_tag} is embedded inside this artwork! Decode it anytime in the **Decode Hidden Message** tab.")

            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.image(st.session_state.last_content.resize((res_choice, res_choice)), caption="Original Content", use_container_width=True)
            with r_col2:
                st.image(st.session_state.last_style.resize((res_choice, res_choice)), caption="Target Style", use_container_width=True)
            with r_col3:
                mode_label = " (Color-Preserved)" if color_preserve else ""
                stego_badge = " 🔒 [Stego]" if st.session_state.get("has_stego", False) else ""
                st.image(st.session_state.last_result, caption=f"Stylized Result (α = {strength}){mode_label}{stego_badge}", use_container_width=True)

            # Download Button (in-memory buffer, lossless PNG)
            buf = io.BytesIO()
            st.session_state.last_result.save(buf, format="PNG")
            file_name = "stylized_stego_artwork.png" if st.session_state.get("has_stego", False) else f"stylized_alpha_{strength:.2f}.png"
            st.download_button(
                label="💾 Download Stylized Artwork (Lossless PNG)",
                data=buf.getvalue(),
                file_name=file_name,
                mime="image/png"
            )

            # Optional On-Demand Metrics
            with st.expander("📊 Quality & Evaluation Metrics (Optional)"):
                if st.button("Compute Metrics for Current Result"):
                    with st.spinner("Calculating SSIM, PSNR, and feature distances..."):
                        model = load_cached_model()
                        metrics = evaluate_transfer_pair(
                            st.session_state.last_result,
                            st.session_state.last_content,
                            st.session_state.last_style,
                            model.encoder
                        )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("SSIM (Structure)", f"{metrics['SSIM (Content Preservation)']:.4f}")
                    m2.metric("PSNR (Reconstruction)", f"{metrics['PSNR (Reconstruction dB)']:.2f} dB")
                    m3.metric("Content Distance", f"{metrics['Content Feature Distance']:.4f}")
                    m4.metric("Style Distance", f"{metrics['Style Feature Distance']:.4f}")

    # Tab 2: Standalone Steganography Decoder
    with tab_decode:
        st.subheader("🕵️ Decode Hidden Message from Artwork")
        st.caption("Upload any artistic image created with this app to extract and reveal its embedded secret message.")

        # Option: Use demo artwork or upload file
        sample_stego_path = os.path.join(SAMPLES_DIR, "stego_sample.png")
        has_sample = os.path.exists(sample_stego_path)

        source_options = ["Upload Image"]
        if has_sample:
            source_options.append("🧪 Try Demo Stego Artwork (Embedded: 'i want to meet u')")

        decode_source = st.radio("Artwork Source", source_options, horizontal=True)

        uploaded_pil = None
        if decode_source == "Upload Image":
            decode_file = st.file_uploader(
                "Upload Stego Artwork (PNG)",
                type=["png", "bmp"],
                help="Note: Secret messages are stored in pixel bit values and require lossless PNG format. JPEG compression removes hidden data.",
                key="decoder_upload"
            )
            if decode_file is not None:
                try:
                    uploaded_pil = Image.open(decode_file).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to open uploaded file: {e}")
                    uploaded_pil = None
        else:
            if has_sample:
                uploaded_pil = Image.open(sample_stego_path).convert("RGB")
                st.info("Loaded demo artwork containing embedded secret message: *'i want to meet u'*. Click below to extract!")

        if uploaded_pil is not None:
            col_dec_img, col_dec_res = st.columns([1, 2])

            with col_dec_img:
                st.image(uploaded_pil, caption="Artwork for Decoding", use_container_width=True)

            with col_dec_res:
                from steganography import is_stego_image, extract_message

                # Instant signature detector
                if is_stego_image(uploaded_pil):
                    st.success("🔒 **Steganographic Signature Detected!** This artwork contains a hidden message.")
                else:
                    st.warning("⚠️ **No Stego Signature Found.** This image does not contain an embedded message, or was converted to JPEG.")

                decode_pwd = st.text_input(
                    "Passcode / Decryption Key (if message was password protected)",
                    type="password",
                    placeholder="Leave blank if unencrypted",
                    key="decoder_pwd"
                )

                if st.button("🔍 Extract Secret Message", type="primary", use_container_width=True):
                    with st.spinner("Analyzing pixel bits and reading steganographic payload..."):
                        pwd = decode_pwd.strip() if decode_pwd.strip() else None
                        success, secret_text, err_msg = extract_message(uploaded_pil, password=pwd)

                    if success:
                        st.success("🎉 **Secret Message Found and Decoded!**")
                        st.text_area("Decoded Secret Message:", value=secret_text, height=160, disabled=True)
                        st.caption(f"Payload Size: {len(secret_text)} characters ({len(secret_text.encode('utf-8'))} bytes)")
                    else:
                        st.error(f"❌ {err_msg}")


if __name__ == "__main__":
    main()

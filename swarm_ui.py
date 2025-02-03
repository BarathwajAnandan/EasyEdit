import streamlit as st
import cv2
import numpy as np
import json
from main import (
    ImageProcessor, 
    process_image_query, 
    execute_function_call
)
import base64
import streamlit.components.v1 as components
import sys
from io import StringIO

def img_to_base64(img):
    """
    Convert an OpenCV image to a base64 string.
    """
    _, buffer = cv2.imencode('.png', img)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 2rem !important;
            text-align: center;
        }
        
        /* Card-like containers */
        .stButton button {
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #4ECDC4 !important;
            border-radius: 10px !important;
            padding: 20px !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa !important;
        }
        
        /* Form styling */
        .stTextInput input {
            border-radius: 8px !important;
            border: 2px solid #e9ecef !important;
            padding: 10px 15px !important;
        }
        
        .stTextInput input:focus {
            border-color: #4ECDC4 !important;
            box-shadow: 0 0 0 2px rgba(78,205,196,0.2) !important;
        }
        
        
        /* Download buttons */
        .download-btn {
            background-color: #4ECDC4 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        
        .download-btn:hover {
            background-color: #45b7b0 !important;
            transform: translateY(-2px) !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa !important;
            border-radius: 8px !important;
        }
        
        /* Success/Error message styling */
        .stSuccess, .stError {
            border-radius: 8px !important;
        }
        
        /* Scrolling text animation */
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        .scrolling-text-container {
            width: 100%;
            overflow: hidden;
            background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 5%, rgba(255,255,255,1) 95%, rgba(255,255,255,0) 100%);
            padding: 10px 0;
            margin-bottom: 20px;
        }
        
        .scrolling-text {
            display: inline-block;
            white-space: nowrap;
            animation: scroll-left 30s linear infinite;
            color: #666;
            font-style: italic;
        }
        
        .command-highlight {
            color: #4ECDC4;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # Custom title with gradient
    st.markdown('<h1 class="main-title">✨ Easy EdIT</h1>', unsafe_allow_html=True)

    # Add scrolling text with example commands
    st.markdown("""
        <div class="scrolling-text-container">
            <div class="scrolling-text">
                Try: 
                <span class="command-highlight">"resize image to 300x300"</span> • 
                <span class="command-highlight">"detect edges"</span> • 
                <span class="command-highlight">"add gaussian blur of 5x5"</span> • 
                <span class="command-highlight">"rotate image 90 degrees"</span> • 
                <span class="command-highlight">"convert to grayscale"</span> • 
                <span class="command-highlight">"what are the dimensions?"</span> • 
                <span class="command-highlight">"crop image from (100,100) to (400,400)"</span> • 
                <span class="command-highlight">"flip image horizontally"</span> • 
                <span class="command-highlight">"flip image vertically"</span> • 
                <span class="command-highlight">"count number of pixels"</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Reset button with custom styling
    if st.button("🔄 Reset Session", 
                type="secondary", 
                help="Clear all images, processing history, and logs",
                key="reset_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'logs' not in st.session_state:
        st.session_state.logs = []

    # Create a custom StringIO object to capture prints
    class StreamCapture(StringIO):
        def write(self, text):
            if text.strip():  # Only capture non-empty strings
                st.session_state.logs.append(text)
            return super().write(text)

    # Capture stdout
    sys.stdout = StreamCapture()
    sys.stderr = StreamCapture()

    # Create a sidebar for processing history
    with st.sidebar:
        st.markdown("""
            <h2 style='
                text-align: center;
                color: #2C3E50;
                padding: 10px;
                border-bottom: 2px solid #4ECDC4;
                margin-bottom: 20px;
            '>🎨 Processing History</h2>
        """, unsafe_allow_html=True)
        
        # Add Step 0 (Initial Image)
        if st.session_state.processor is not None:
            with st.expander("Step 0 (Initial Image)"):
                st.markdown("**Original Image**")
                # Show initial image shape
                st.markdown(f"**Shape:** {st.session_state.processor.initial_image.shape}")
                # Show thumbnail of initial image
                st.image(
                    cv2.cvtColor(st.session_state.processor.initial_image, cv2.COLOR_BGR2RGB),
                    caption="Initial Image",
                    width=200
                )
                # Add checkout button for initial state
                if st.button("Reset to Original", key="checkout_0"):
                    # Reset processor to initial state
                    initial_image = st.session_state.processor.initial_image.copy()
                    st.session_state.processor = ImageProcessor(initial_image)
                    st.rerun()
                st.divider()

        # Display subsequent processing steps
        if st.session_state.processor and st.session_state.processor.processing_steps:
            for i, step in enumerate(st.session_state.processor.processing_steps, 1):
                with st.expander(f"Step {i}"):
                    # Show function name
                    st.markdown(f"**Function:** `{step['function_call']['function_name']}`")
                    # Show parameters
                    st.markdown("**Parameters:**")
                    params = step['function_call']['parameters']
                    for j, param in enumerate(params):
                        if j == 0 and param == 'image':  # Skip showing 'image' parameter
                            continue
                        st.code(param)
                    
                    # Show image shape after operation
                    if step['image_shape']:
                        st.markdown(f"**Output Shape:** {step['image_shape']}")
                    
                    # Show thumbnail of processed image
                    if 'image' in step:
                        st.image(
                            cv2.cvtColor(step['image'], cv2.COLOR_BGR2RGB),
                            caption=f"Result after step {i}",
                            width=200  # Smaller thumbnail size
                        )
                    
                    # Add checkout button
                    if st.button(f"Checkout Step {i}", key=f"checkout_{i}"):
                        # Restore the image state to this step
                        st.session_state.processor.current_image = step['image'].copy()
                        # Truncate the processing steps to this point
                        st.session_state.processor.processing_steps = st.session_state.processor.processing_steps[:i]
                        # Update the metadata history to reflect the truncated steps
                        st.session_state.processor.metadata["history"] = st.session_state.processor.metadata["history"][:i]
                        st.rerun()
                    st.divider()

    # Main content area before file uploader
    uploaded_file = st.file_uploader(
         "📤 Upload your image",
         type=['jpg', 'jpeg', 'png'],
         help="Supported formats: JPG, JPEG, PNG"
    )

    # Store uploaded image in session state when a new file is uploaded
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if st.session_state.uploaded_image is None:
            st.session_state.uploaded_image = image
            st.session_state.processor = ImageProcessor(image)

    # Input image section
    st.subheader("Input Image")
    if st.session_state.uploaded_image is not None:
        # Get image dimensions from the processor's initial image
        h, w = st.session_state.processor.initial_image.shape[:2]
        img_base64 = img_to_base64(st.session_state.processor.initial_image)
        img_id = "clickable-image"
        coord_id = "coords-display"
        html_code = f"""
        <div style="position: relative; display: inline-block; margin-bottom: 0;">
            <img id="{img_id}" src="data:image/png;base64,{img_base64}" 
                 style="width:100%; cursor: crosshair; border:1px solid #ddd; border-radius:4px; display:block;">
            <div id="{coord_id}" style="
                position: absolute;
                background: rgba(255,0,0,0.7);
                color: white;
                padding: 5px 5px;
                border-radius: 2px;
                display: none;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
            ">
                Coordinates: (x, y)
            </div>
        </div>
        <div style="margin-top: 5px; margin-bottom: 0; display: flex; align-items: center; gap: 10px;">
            <label style="color: #555; font-size: 14px;"> Click image to see x and y:</label>
            <input type="text" id="copyable-coords" readonly 
                   style="width: 100px;
                          padding: 3px 8px;
                          border: 1px solid #ddd;
                          border-radius: 4px;
                          font-family: monospace;
                          color: #444;
                          background: #f8f9fa;"
                   value="(x, y)">
        </div>
        <script>
            const img = document.getElementById('{img_id}');
            const coordsDisplay = document.getElementById('{coord_id}');
            const copyableInput = document.getElementById('copyable-coords');
            const maxWidth = {w};
            const maxHeight = {h};

            img.addEventListener('click', function(event) {{
                const rect = img.getBoundingClientRect();
                const scaleX = {w} / rect.width;
                const scaleY = {h} / rect.height;
                
                // Calculate coordinates and clamp them to image bounds
                let x = Math.round((event.clientX - rect.left) * scaleX);
                let y = Math.round((event.clientY - rect.top) * scaleY);
                
                // Ensure coordinates stay within image bounds
                x = Math.max(0, Math.min(x, maxWidth - 1));
                y = Math.max(0, Math.min(y, maxHeight - 1));

                const coordText = '(' + x + ', ' + y + ')';
                coordsDisplay.textContent = coordText;
                copyableInput.value = coordText;  // Only the coordinates, no label
                
                // Position the coordinate display
                coordsDisplay.style.left = (event.clientX - rect.left + 10) + 'px';
                coordsDisplay.style.top = (event.clientY - rect.top - 20) + 'px';
                coordsDisplay.style.display = 'block';
            }});

            // Hide the coordinates when clicking outside the image
            document.addEventListener('click', function(e) {{
                if (!img.contains(e.target)) {{
                    coordsDisplay.style.display = 'none';
                }}
            }});
        </script>
        """
        # Calculate a more precise height for the component
        coord_input_height = 500  # Height for the coordinates input area
        components.html(html_code, height=coord_input_height, scrolling=False)
    else:
        st.info("Upload an image to begin")

    # Add minimal spacing
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Output image section
    st.subheader("Processed Image")
    if st.session_state.processor is not None:
        st.image(
            cv2.cvtColor(st.session_state.processor.current_image, cv2.COLOR_BGR2RGB),
        )
        
        # Modify the download buttons section
        st.markdown('<div style="display: flex; justify-content: center; gap: 20px;">', unsafe_allow_html=True)
        col_png, col_jpg, col_pdf = st.columns(3)
        
        with col_png:
            _, png_buffer = cv2.imencode('.png', st.session_state.processor.current_image)
            st.download_button(
                label="📥 Download PNG",
                data=png_buffer.tobytes(),
                file_name="processed_image.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_jpg:
            _, jpg_buffer = cv2.imencode('.jpg', st.session_state.processor.current_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            st.download_button(
                label="📥 Download JPG",
                data=jpg_buffer.tobytes(),
                file_name="processed_image.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        
        with col_pdf:
            try:
                from PIL import Image
                from io import BytesIO
                
                # Convert OpenCV image to PIL Image
                img_rgb = cv2.cvtColor(st.session_state.processor.current_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Create PDF in memory
                pdf_buffer = BytesIO()
                pil_image.save(pdf_buffer, format='PDF', resolution=100.0)
                pdf_bytes = pdf_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name="processed_image.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF conversion failed: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Process an image to see the result")

    # Query form with enhanced styling
    if st.session_state.uploaded_image is not None:
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        with st.form(key='query_form', clear_on_submit=False):
            query = st.text_input(
                "🔍 What would you like to do with your image?",
                placeholder="Example: resize image to 100x100 or what is the width of this image?",
                value=""
            )
            submit = st.form_submit_button("✨ Process Image", type="primary")
            
            if submit and query:
                try:
                    function_call_params = process_image_query(
                        st.session_state.processor.current_image,
                        query,
                        st.session_state.processor
                    )
                    
                    output = execute_function_call(st.session_state.processor, function_call_params, query)
                    
                    # Handle different types of results
                    if function_call_params.get("type") == "analysis":
                        # Create containers for analysis results
                        result_container = st.container()
                        with result_container:
                            # Show the interpreted result prominently
                            st.success(output['interpretation'])
                            
                            # Show technical details in an expander
                            with st.expander("Technical Details"):
                                st.markdown("**Raw Result:**")
                                st.code(str(output['raw_result']))
                                st.markdown("**Function Used:**")
                                st.code(function_call_params['function_name'])
                    else:
                        # For edit operations, rerun to update the UI
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
            elif submit and not query:
                st.warning("Please enter a request")

        # Show a placeholder for analysis results when no query is submitted
        if 'analysis_result' not in st.session_state and not submit:
            st.info("Enter an analysis query (e.g., 'what are the dimensions?') or an edit request (e.g., 'resize to hxw')")
        st.markdown('</div>', unsafe_allow_html=True)

    # Debug logs with enhanced styling
    with st.expander("🐛 Debug Logs", expanded=False):
        # Create a container for logs
        log_container = st.container()
        
        # Show logs in reverse chronological order (newest first)
        with log_container:
            for log in reversed(st.session_state.logs):
                if "error" in log.lower() or "exception" in log.lower():
                    st.error(log)
                else:
                    st.text(log)
        
        # Add a clear logs button
        if st.button("Clear Logs"):
            st.session_state.logs = []
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__ 
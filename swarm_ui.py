import streamlit as st
import cv2
import numpy as np
import json
from main import (
    ImageProcessor, 
    process_image_query, 
    execute_function_call
)

def main():
    st.title("Easy EdIT")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = None

    # File upload at the top
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )

    # Create two columns for input and output images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        if uploaded_file:
            # Load and display input image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Initialize processor
            if st.session_state.processor is None:
                st.session_state.processor = ImageProcessor(image)
            
            # Display input image
            st.image(
                cv2.cvtColor(st.session_state.processor.initial_image, cv2.COLOR_BGR2RGB),
            )
        else:
            st.info("Upload an image to begin")

    with col2:
        st.subheader("Processed Image")
        if st.session_state.processor is not None:
            st.image(
                cv2.cvtColor(st.session_state.processor.current_image, cv2.COLOR_BGR2RGB),
            )
            # Download button for processed image
            _, buffer = cv2.imencode('.png', st.session_state.processor.current_image)
            st.download_button(
                label="Download Processed Image",
                data=buffer.tobytes(),
                file_name="processed_image.png",
                mime="image/png"
            )
        else:
            st.info("Process an image to see the result")

    # Query input and process button below the images
    if uploaded_file:
        query = st.text_input(
            "Enter your image processing request:",
            placeholder="Example: resize image to 100x100"
        )

        if st.button("Process Image", type="primary"):
            if query:
                with st.spinner("Processing..."):
                    try:
                        # Process query
                        function_call_params = process_image_query(
                            st.session_state.processor.current_image, 
                            query
                        )
                        
                        # Execute function
                        processed_image = execute_function_call(
                            st.session_state.processor, 
                            function_call_params
                        )

                        if processed_image is not None:
                            st.success("Processing complete!")
                            # Force rerun to update the UI
                            st.rerun()
                        else:
                            st.error("Processing failed")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a processing request")

if __name__ == "__main__":
    main() 
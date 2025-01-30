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
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    # Create a sidebar for processing history
    with st.sidebar:
        st.title("Processing History")
        
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
                        st.rerun()
                    
                    st.divider()

    # File upload at the top
    uploaded_file = st.file_uploader(
        "Upload an image",
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

    # Create two columns for input and output images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        if st.session_state.uploaded_image is not None:
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
    if st.session_state.uploaded_image is not None:
        with st.form(key='query_form', clear_on_submit=False):
            query = st.text_input(
                "Enter your image processing request:",
                placeholder="Example: resize image to 100x100",
                value="resize image to 100x100"  # Added default value
            )
            # Hidden submit button for Enter key
            submit = st.form_submit_button("Submit", type="primary")
            # Hide the submit button using CSS
            st.markdown(
                """
                <style>
                    button[kind="primary"] {
                        display: none;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            
        # Visible process button (outside form)
        process_button = st.button("Process Image", type="primary")
        
        # Process if either Enter was pressed or button was clicked
        if submit or process_button:
            if query:
                with st.spinner("Processing..."):
                    try:
                        # Process query
                        function_call_params = process_image_query(
                            st.session_state.processor.current_image, 
                            query,
                            processor=st.session_state.processor
                        )
                        
                        # Execute function
                        processed_image = execute_function_call(
                            st.session_state.processor, 
                            function_call_params
                        )

                        if processed_image is not None:
                            st.success("Processing complete!")
                            st.rerun()
                        else:
                            st.error("Processing failed")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a processing request")

if __name__ == "__main__":
    main() 
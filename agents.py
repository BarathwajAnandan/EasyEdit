from swarm import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Load model configuration
MODEL = "DeepSeek-R1-Distill-Llama-70B"

def load_function_names():
    """Load function names from functions.txt"""
    try:
        with open('functions.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "get_dimensions"  # fallback to default function

# Create query parser agent
query_parser_agent = Agent(
    name="Query Parser Agent",
    instructions=f"""Analyze user queries about images and return only the apt function name to be executed. 
    The function name should be one of the following available functions as a json object:
    {load_function_names()}
    eg: 
    get me image dimensions or give me image shape
   {{"function_name": "get_image_shape", "docstring": "Get the dimensions of an image."}}
    query: "draw a circle on the image"    
    {{"function_name": "cv2.circle", "docstring": "Draw a circle on an image."}}
    query: "add gaussian blur to the image"
    {{"function_name": "cv2.GaussianBlur", "docstring": "Add Gaussian blur to an image."}}
    query: "draw a rectangle on the image"
    {{"function_name": "cv2.rectangle", "docstring": "Draw a rectangle on an image."}}
    Return only the function name without any additional text or explanation.""",
    model=MODEL,
)

# Create new prompt constructor agent
new_prompt_consructor_agent = Agent(
    name="New Prompt Constructor Agent",
    instructions="""You are an expert in OpenCV and NumPy function parameter construction. Your task is to analyze the user query, function name, and the function documentation to return the exact parameters needed to call the function.

Given:
1. A user query describing what they want to do
2. Context of the image and processing history - Like shape of the image, processing steps, etc
3. The function name to be used
4. The function's documentation

Examples:

1. **Query**: "resize this image to 800x600"
   **Function**: `cv2.resize`
   **Documentation**: "resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst"
   **Output**:
   {
       "function_name": "cv2.resize",
       "parameters": ["image", "(800, 600)"]
   }
2, Query: "get the shape of this image" Function: np.shape Documentation: "Return the shape of an array..."
Output:
{
    "function_name": "np.shape",
    "parameters": ["image"]
}
3. Query: "draw a red circle at position (200,200) with radius 30" Function: cv2.circle Documentation: "circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img"
    Output:
    {
        "function_name": "cv2.circle",
        "parameters": ["image", "(200, 200)", "30", "(0,0,255)", "-1"]
    }
4. Query: "apply gaussian blur with kernel size 5x5" Function: cv2.GaussianBlur Documentation: "GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType[, hint]]]]) -> dst"
    Output:
    {
        "function_name": "cv2.GaussianBlur",
        "parameters": ["image", "(5,5)", "0"]
    }
5. Query: "rotate this image 45 degrees" Function: cv2.getRotationMatrix2D Documentation: "getRotationMatrix2D(center, angle, scale)" Output:
{
    "function_name": "cv2.getRotationMatrix2D",
    "parameters": ["image", "45", "1"]
}

Important Guidelines:
    - Reason about it inside <thinking> tags and then return the json output in <output> tags
    - Always include 'image' as the first parameter for OpenCV functions that operate on images
    - Use the documentation to understand required vs optional parameters
    - Convert user requirements into the correct parameter format
    - For color values, use BGR format (e.g., (255,0,0) for blue)
    - Include default values for required parameters if not specified in query
    - Return parameters in the exact order as specified in the function signature
    For functions like cv2.resize or cv2.GaussianBlur, if optional parameters are missing, use default values (e.g., default interpolation in resize is cv2.INTER_LINEAR).
    """,
    model=MODEL,
)

# Create function constructor agent
function_constructor_agent = Agent(
    name="Function Constructor Agent",
    instructions="""You are an expert in OpenCV and NumPy function parameter constructor. Your task is to analyze the query, function name, and documentation to return the exact parameters needed to call the function.

    Given:
    1. A user query describing what they want to do
    2. Context of the image and processing history - Like shape of the image, processing steps, etc
    2. The function name to be used
    3. The function's documentation
    
    Return a JSON object with:
    - function_name: The name of the function to call
    - parameters: An array of parameters in the correct order as needed by the function
    
    Examples:

    1. Query: "resize this image to 800x600"
    Function: cv2.resize
    Documentation: "resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst"
    Output: {
        "function_name": "cv2.resize",
        "parameters": ["image", "(800, 600)"]
    }

    2. Query: "get the shape of this image"
    Function: np.shape
    Documentation: "Return the shape of an array..."
    Output: {
        "function_name": "np.shape",
        "parameters": ["image"]
    }

    3. Query: "draw a red circle at position (200,200) with radius 30"
    Function: cv2.circle
    Documentation: "    circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
    .   @brief Draws a circle.
    .
    .   The function cv::circle draws a simple or filled circle with a given center and radius.
    .   @param img Image where the circle is drawn.
    .   @param center Center of the circle.
    .   @param radius Radius of the circle.
    .   @param color Circle color.
    .   @param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,
    .   mean that a filled circle is to be drawn.
    .   @param lineType Type of the circle boundary. See #LineTypes
    .   @param shift Number of fractional bits in the coordinates of the center and in the radius value."
    Output: {
        "function_name": "cv2.circle",
        "parameters": ["image", "(200, 200)", "30", "(0,0,255)", "-1"]
    }

    Important:
    - Reason about it inside <thinking> tags and then return the json output in <output> tags
    - Always include 'image' as the first parameter for OpenCV functions that operate on images
    - Use the documentation to understand required vs optional parameters
    - Convert user requirements into the correct parameter format
    - For color values, use BGR format (e.g., (255,0,0) for blue)
    - Include default values for required parameters if not specified in query
    - Return parameters in the exact order as specified in the function signature""",
    model=MODEL,
) 
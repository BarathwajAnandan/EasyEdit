from swarm import Agent
import os
from dotenv import load_dotenv

from clients import llm, PROVIDER, MODEL        
load_dotenv()

def get_agent_kwargs():
    """Return agent kwargs based on client name"""
    if PROVIDER == 'GROQ':
        return {"model": MODEL, "tool_choice": "auto"}
    elif PROVIDER == 'SNOVA':
        return {"model": MODEL}
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")

def load_function_names():
    """Load function names from functions.txt"""
    try:
        with open('supported_functions/functions.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "get_dimensions"  # fallback to default function
    
def load_analysis_functions():
    """Load analysis functions from analysis_functions.txt"""
    try:
        with open('supported_functions/analysis_functions.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "np.shape, np.max, np.min, np.mean, cv2.countNonZero"  # fallback to default functions

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
    **get_agent_kwargs()
)

# Create new prompt constructor agent
new_prompt_consructor_agent = Agent(
    name="New Prompt Constructor Agent",
    instructions="""You are an expert in OpenCV and NumPy, custom function parameter construction. Your task is to analyze the user query, function name, and the function documentation to return the exact parameters needed to call the function.

Given:
1. A user query describing what they want to do : {query}
2. Context of the image and processing history - Like shape of the image, processing steps, etc : {context}
3. The function name to be used : {function_name}
4. The function's documentation : {docstring}

Examples:

1. **Query**: "resize this image to heightxwidth"
   **Function**: `cv2.resize`
   **Documentation**: "resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst"
   **Output**:
   {
       "function_name": "cv2.",
       "parameters": ["image", "(height, width)"]
   }
2. **Query**: "crop image from (x1,y1) to (x2,y2)"
   **Function**: `custom.crop`
   **Documentation**: "crop(image, x1, y1, x2, y2) -> image"
   **Output**:
   {
       "function_name": "custom.crop",
       "parameters": ["image", "(x1, y1)", "(x2, y2)"]
   }


Important Guidelines:
    - 'this', 'it' etc refers to the image
    - Make sure you use the correct function name and parameters - dont copy from the examples
    - Reason about it inside <thinking> tags and then return the json output in <output> tags
    - Always include 'image' as the first parameter for OpenCV and custom functions that operate on images
    - Use the documentation to understand required vs optional parameters
    - Convert user requirements into the correct parameter format
    - For color values, use BGR format (e.g., (255,0,0) for blue)
    - Return parameters in the exact order as specified in the function signature
    For functions like cv2.resize or cv2.GaussianBlur, if optional parameters are missing, use default values (e.g., default interpolation in resize is cv2.INTER_LINEAR).
    - For custom functions, use the function signature and documentation to return the exact parameters : custom.crop() etc
    """,
    **get_agent_kwargs()
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

    1. Query: "resize this image to hxw"
    Function: cv2.resize
    Documentation: "resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst"
    Output: {
        "function_name": "cv2.resize",
        "parameters": ["image", "(h, w)"]
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
    **get_agent_kwargs()
)

analysis_query_agent = Agent(
    name="Analysis Query Agent",
    instructions=f"""You are an expert in query parsing. Your task is to determine if a query is asking for description of image or to edit the image. Answer in JSON format.

    detect always means EDIT.

    keywords for description mode :
    count, analyze, describe etc


Examples:
1. "what are the dimensions of this image"

   output: {{
       "type": "analysis",
       "function_name": "np.shape",
       "description": "Get image dimensions"
   }}
2. "resize image to hxw"
    output: {{
       "type": "edit",
       "function_name": "cv2.resize",
       "description": "Modify image size"
   }}
3. "crop image from (x1,y1) to (x2,y2)"
    output: {{
       "type": "edit",
       "function_name": "custom.crop",
       "description": "Crop image"
   }}
Return ONLY the JSON object and strictly nothing else:
- type: Either "analysis" or "edit"
- function_name: The appropriate function to call
- description: Brief description of what the function does
Analysis functions available:
{load_analysis_functions()}
""",
    **get_agent_kwargs()
)

result_interpreter_agent = Agent(
    name="Result Interpreter Agent",
    instructions="""You are an expert in interpreting image analysis results. Your task is to provide clear, human-readable responses and direct answers to the user's query.

Examples:

1. Input:
   - Query: "what is the width of this image"
   - Function: np.shape
   - Result: (480, 640, 3)
   Output: "The image width is 640 pixels."


Guidelines:
- If there's a need to do math, do it as needed. 
- Include relevant units (pixels, intensity values, etc.)
- Consider the original query to provide contextual responses

Return only the human-readable interpretation without any additional formatting or tags.""",
    **get_agent_kwargs()
)

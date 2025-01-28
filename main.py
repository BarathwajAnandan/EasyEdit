from swarm import Swarm, Agent
import os
from dotenv import load_dotenv
from openai import OpenAI
import cv2
import numpy as np
import json
import re
load_dotenv()

client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=os.getenv("SNOVA_API_KEY"))
client = Swarm(client)

# At the top level, add a list to store processing steps
processing_steps = []

def get_dimensions(image: np.ndarray) -> str:
    """Returns image dimensions in (width, height) format"""
    height, width = image.shape[:2]
    return f"Image dimensions: {width}x{height}"

def transfer_to_cv_agent():
    """Handles transfer to computer vision function agent"""
    return cv_function_agent

# # Create computer vision function agent
# cv_function_agent = Agent(
#     name="CV Function Agent",
#     instructions="You are an expert in OpenCV image processing operations. Analyze the query and execute the appropriate function.",
#     functions=[get_dimensions],
#     model="gpt-4o"
# )

def load_function_names():
    """Load function names from functions.txt"""
    try:
        with open('functions.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "get_dimensions"  # fallback to default function

# Create query parser agent
# function_names = load_function_names()
# print(function_names)
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
    model="Meta-Llama-3.3-70B-Instruct"
)

function_constructor_agent = Agent(
    name="Function Constructor Agent",
    instructions="""You are an expert in OpenCV and NumPy function parameter constructor. Your task is to analyze the query, function name, and documentation to return the exact parameters needed to call the function.

    Given:
    1. A user query describing what they want to do
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

    4. Query: "apply gaussian blur with kernel size 5x5"
    Function: cv2.GaussianBlur
    Documentation: "    GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType[, hint]]]]) -> dst
    .   @brief Blurs an image using a Gaussian filter.
    .
    .   The function convolves the source image with the specified Gaussian kernel. In-place filtering is
    .   supported.
    .
    .   @param src input image; the image can have any number of channels, which are processed
    .   independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    .   @param dst output image of the same size and type as src.
    .   @param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
    .   positive and odd. Or, they can be zero's and then they are computed from sigma.
    .   @param sigmaX Gaussian kernel standard deviation in X direction.
    .   @param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
    .   equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
    .   respectively (see #getGaussianKernel for details); to fully control the result regardless of
    .   possible future modifications of all this semantics, it is recommended to specify all of ksize,
    .   sigmaX, and sigmaY.
    .   @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
    .   @param hint Implementation modfication flags. See #AlgorithmHint
    .
    .   @sa  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur"
    Output: {
        "function_name": "cv2.GaussianBlur",
        "parameters": ["image", "(5,5)", "0"]
    }

    Important:
    - Reason about it inside <thinking> tags and then return the json output in <output> tags
    - Always include 'image' as the first parameter for OpenCV functions that operate on images
    - Use the documentation to understand required vs optional parameters
    - Convert user requirements into the correct parameter format
    - For color values, use BGR format (e.g., (255,0,0) for blue)
    - Include default values for required parameters if not specified in query
    - Return parameters in the exact order as specified in the function signature
    """,

    model="Meta-Llama-3.3-70B-Instruct"
)

def get_function_info(function_name: str) -> dict:
    """Get the signature and docstring for a specific OpenCV function"""
    try:
        lib, fn = function_name.split(".")  # e.g. cv2.circle
        print("getting function info for: ", fn, "from library: ", lib)
        if lib == 'np':
            fn = getattr(np, fn)
        elif lib == 'cv2':
            fn = getattr(cv2, fn)
        else:
            raise ValueError(f"Invalid library: {lib}")
        return {
            "name": function_name,
            "docstring": str(fn.__doc__),
        }
    except AttributeError:
        return {
            "name": function_name,
            "docstring": "Function not found in cv2",
        }

def construct_function_call(function_info: dict) -> dict:
    """
    Use function_constructor_agent to determine the exact parameters needed for the function call
    """
    response = client.run(
        agent=function_constructor_agent,
        messages=[{
            "role": "user",
            "content": f"""Query: {function_info['query']}
Function: {function_info['name']}
Documentation: {function_info['docstring']}

Please provide the function parameters based on the query and documentation."""
        }],
    )
    
    response_message = response.messages[-1]["content"]
    try:
        # Extract JSON from between <output> tags
        json_str = re.search(r'<output>\s*(.*?)\s*</output>', response_message, re.DOTALL)
        if json_str:
            params = json.loads(json_str.group(1))
            return params
        else:
            print("Warning: Could not find <output> tags in response:", response_message)
            return {
                "function_name": function_info['name'],
                "parameters": ["image"]  # fallback to basic parameter
            }
    except json.JSONDecodeError:
        print("Warning: Could not parse agent response as JSON:", response_message)
        return {
            "function_name": function_info['name'],
            "parameters": ["image"]  # fallback to basic parameter
        }

def process_image_query(image: np.ndarray, query: str):
    # Get function name from query parser
    response = client.run(
        agent=query_parser_agent,
        messages=[{
            "role": "user", 
            "content": query,
        }],
    )
    response_message = response.messages[-1]["content"]
    import json
    function_name = json.loads(response_message)["function_name"]
    assert function_name in load_function_names(), "Function name not found in available functions"
    # print(query, "function_name: ", function_name)
    
    # Get function information and documentation
    function_info = get_function_info(function_name)
    function_info['query'] = query
    # print(f"\nFunction Information:")
    # print(f"Name: {function_info['name']}")
    # print(f"Documentation: {function_info['docstring']}...")
    
    # Construct function parameters
    function_call = construct_function_call(function_info)
    # print("\nConstructed Function Call:")
    # print(json.dumps(function_call, indent=2))
    
    return function_call

class ImageProcessor:
    def __init__(self, initial_image: np.ndarray):
        self.processing_steps = []
        self.current_image = initial_image.copy()
        self.initial_image = initial_image.copy()
    
    def add_step(self, function_call_params: dict, processed_image: np.ndarray):
        """Add a processing step and update current image"""
        self.processing_steps.append({
            "function_call": function_call_params,
            "image_shape": processed_image.shape if processed_image is not None else None
        })
        self.current_image = processed_image.copy()
    
    def get_current_state(self):
        """Get current processing state"""
        return {
            "steps": self.processing_steps,
            "current_image": self.current_image,
            "initial_image_shape": self.initial_image.shape,
            "current_image_shape": self.current_image.shape
        }

def execute_function_call(processor: ImageProcessor, function_call_params: dict):
    """
    Dynamically execute the function with given parameters
    """
    try:
        # Split library and function name
        lib_name, func_name = function_call_params['function_name'].split('.')
        
        # Get the appropriate library
        lib = {'cv2': cv2, 'np': np}[lib_name]
        
        # Get the function from the library
        func = getattr(lib, func_name)
        
        # Convert string parameters to actual Python objects
        processed_params = []
        for param in function_call_params['parameters']:
            if param == 'image':
                processed_params.append(processor.current_image)
            else:
                # Safely evaluate string parameters (tuples, numbers, etc)
                try:
                    processed_params.append(eval(param))
                except:
                    processed_params.append(param)
        
        # Execute the function with the processed parameters
        processed_image = func(*processed_params)
        
        # Add step and update current image
        processor.add_step(function_call_params, processed_image)
        
        return processed_image
        
    except Exception as e:
        print(f"Error executing function: {str(e)}")
        return None

if __name__ == "__main__":
    # Test with sample image and query
    queries = [
        "resize image to 100x100",
        "draw a red circle at position left corner of the image with radius 30",
        "add gaussian blur with kernel size 11x11"
    ]

    sample_image = cv2.imread('../test.png')
    processor = ImageProcessor(sample_image)

    for query in queries:
        function_call_params = process_image_query(processor.current_image, query)
        print("Query:", query)
        print("Function call params:", function_call_params)
        
        processed_image = execute_function_call(processor, function_call_params)
        if processed_image is not None:
            cv2.imwrite('processed_image.png', processor.current_image)
        print("-" * 50)

    # Print all processing steps at the end
    print("\nProcessing Steps:")
    state = processor.get_current_state()
    for i, step in enumerate(state["steps"], 1):
        print(f"Step {i}:")
        print("Function:", json.dumps(step["function_call"], indent=2))
        # print("Image shape after step:", step["image_shape"])
        print()





    # print(result)  # Output: "Image dimensions: 640x480"
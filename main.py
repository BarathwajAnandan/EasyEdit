from swarm import Swarm, Agent
import os
from dotenv import load_dotenv
from openai import OpenAI
import cv2
import numpy as np
import json
import re
import base64
from agents import (
    query_parser_agent,
    new_prompt_consructor_agent,
    function_constructor_agent,
    load_function_names,
    load_analysis_functions,
    analysis_query_agent,
    result_interpreter_agent,
)

load_dotenv()

# client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=os.getenv("SNOVA_API_KEY"))
client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.getenv('GROQ_API_KEY'),
)
MODEL = 'deepseek-r1-distill-llama-70b'
# MODEL = 'llama3-70b-8192'
# MODEL = "Meta-Llama-3.3-70B-Instruct"
# MODEL = "DeepSeek-R1-Distill-Llama-70B"
client = Swarm(client)

# At the top level, add a list to store processing steps
processing_steps = []


# # Create computer vision function agent
# cv_function_agent = Agent(
#     name="CV Function Agent",
#     instructions="You are an expert in OpenCV image processing operations. Analyze the query and execute the appropriate function.",
#     functions=[get_dimensions],
#     model="gpt-4o"
# )

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
    context = function_info['context']
    print("CONTEXT: ", context)
    print("INSIDE CONSTRUCT FUNCTION CALL")
    
    response = client.run(
        agent=new_prompt_consructor_agent, #change to deepthinking_function_constructor_agent for deepthinking (WIP)
        messages=[{
            "role": "user",
            "content": f"""Context: {context}
Query: {function_info['query']}
Function: {function_info['name']}
Documentation: {function_info['docstring']}

Please provide the function parameters based on the query and documentation."""
        }],
    )
    
    response_message = response.messages[-1]["content"]
    print("constructing function call response... ")
    # if '<think>' in response_message:
    #     params = extract_json_from_string(response_message)
    #     return params
    try:
        # Extract JSON from between <output> tags
        json_str = re.search(r'<output>\s*(.*?)\s*</output>', response_message, re.DOTALL)
        if json_str:
            params = json.loads(json_str.group(1))
            breakpoint()
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
            "json_str": json_str,
            "function_name": function_info['name'],
            "parameters": ["image"]  # fallback to basic parameter
        }



class ImageProcessor:
    def __init__(self, initial_image: np.ndarray):
        self.processing_steps = []
        self.current_image = initial_image.copy()
        self.initial_image = initial_image.copy()
        # Initialize metadata with correct shape format
        h, w = initial_image.shape[:2]  # Get height and width
        self.metadata = {
            "initial_shape": f"(width={w}, height={h})",  # CV2 format (w,h)
            "current_shape": f"(width={w}, height={h})",
            "history": [],
            "dtype": str(initial_image.dtype)
        }
    
    def add_step(self, function_call_params: dict, processed_image: np.ndarray):
        """Add a processing step and update current image and metadata"""
        # Create description of the change
        func_name = function_call_params['function_name']
        params = function_call_params['parameters']
        param_str = ', '.join([p for p in params if p != 'image'])
        change_description = f"Applied {func_name} with parameters: {param_str}"
        
        # Get shape in CV2 format (width, height)
        h, w = processed_image.shape[:2]
        shape_str = f"(width={w}, height={h})"
        
        # Update metadata
        self.metadata["current_shape"] = shape_str
        self.metadata["history"].append({
            "step": len(self.processing_steps) + 1,
            "operation": change_description,
            "shape": shape_str
        })
        
        # Add step to processing history
        self.processing_steps.append({
            "function_call": function_call_params,
            "image_shape": shape_str,
            "image": processed_image.copy() if processed_image is not None else None
        })
        
        self.current_image = processed_image.copy()
    
    def get_current_state(self):
        """Get current processing state including metadata"""
        return {
            "steps": self.processing_steps,
            "current_image": self.current_image,
            "initial_image_shape": self.initial_image.shape,
            "current_image_shape": self.current_image.shape,
            "metadata": self.metadata
        }
    
    def get_context_string(self) -> str:
        """Get a formatted string of metadata for model context"""
        context = [
            f"Image Information:",
            f"- Initial shape: {self.metadata['initial_shape']}",
            f"- Current shape: {self.metadata['current_shape']}",
            f"- Data type: {self.metadata['dtype']}",
            f"\nProcessing History:"
        ]
        
        for step in self.metadata["history"]:
            context.append(
                f"Step {step['step']}: {step['operation']}\n"
                f"  Shape changed from {step['shape']}"
            )
        
        return "\n".join(context)
    
    def initial_image_to_base64(self):
        """Convert initial image to base64 for HTML display"""
        _, buffer = cv2.imencode('.png', self.initial_image)
        return base64.b64encode(buffer).decode('utf-8')

def extract_json_from_string(text: str) -> dict:
    """
    Extracts a JSON object from a string, specifically looking for content after a <think> tag.

    Args:
        text: The input string containing the JSON object.

    Returns:
        A dictionary representing the extracted JSON object, or None if no valid JSON is found.
    """
    # Use regex to find content between </think> and the next {

    if '<think>' in text:
        match = re.search(r'</think>\s*({.*})', text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))

        except json.JSONDecodeError:
            print("Warning: Could not parse extracted content as JSON.")
            return None
    else:
        print("Warning: Could not find JSON after </think> tag.")
        return None

def analyze_image(image: np.ndarray, function_name: str) -> str:
    """
    Analyze image using the specified function and return a human-readable result
    """
    try:
        # Split library and function name
        lib_name, func_name = function_name.split('.')
        
        # Get the appropriate library
        lib = {'cv2': cv2, 'np': np}[lib_name]
        
        # Get the function
        func = getattr(lib, func_name)
        
        # Execute the function
        result = func(image)
        
        # Format the result based on function
        if function_name == 'np.shape':
            return f"Image dimensions: {result[1]}x{result[0]} (width x height)"
        elif function_name in ['np.max', 'np.min']:
            return f"{func_name.capitalize()} pixel value: {result}"
        elif function_name == 'np.mean':
            return f"Average pixel value: {result:.2f}"
        elif function_name == 'cv2.countNonZero':
            return f"Number of non-zero pixels: {result}"
        else:
            return f"Analysis result: {result}"
            
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def parse_analysis_response(response: str) -> dict:
    """Parse the JSON response from analysis_query_agent.
    
    Args:
        response (str): The raw response string containing JSON, either as:
            - Plain JSON: {"type": "analysis", ...}
            - Markdown code block: ```json\n{"type": "analysis", ...}```
            
    Returns:
        dict: Parsed JSON object with keys 'type', 'function_name', and 'description'
    """
    import json
    # Remove any leading/trailing whitespace and newlines
    response = response.strip()
    
    # If response is wrapped in backticks, extract just the JSON part
    if '```' in response:
        # Split by ``` and take the middle part
        parts = response.split('```')
        # Get the part that contains the JSON (usually the second part)
        json_str = [part for part in parts if '{' in part][0]
        # If there's a language identifier (like 'json\n'), remove it
        if 'json\n' in json_str:
            json_str = json_str.split('\n', 1)[1]
    else:
        # Response is plain JSON
        json_str = response
    
    # Remove any remaining whitespace and newlines
    json_str = json_str.strip()
    
    # Parse the cleaned JSON string
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")

def process_image_query(image: np.ndarray, query: str, processor: ImageProcessor = None):
    """Modified to handle both analysis and editing queries"""
    # Get context from processor if available
    context = ""
    if processor:
        context = processor.get_context_string()
    
    # First, determine if this is an analysis or edit query
    response = client.run(
        agent=analysis_query_agent,
        messages=[{
            "role": "user", 
            "content": f"Query: {query}\nContext: {context}"
        }],
    )
    query_type = parse_analysis_response(response.messages[-1]["content"])
    if query_type["type"] == "analysis":
        # Handle analysis query
        function_name = query_type["function_name"]
        assert function_name in load_analysis_functions(), "Function name not found in available functions"

        function_info = get_function_info(function_name)
        function_info['query'] = query
        function_info['context'] = context

        function_call = construct_function_call(function_info)
        function_call["type"] = "analysis"
        return function_call
    else:
        # Handle edit query (existing logic)
        function_name = query_type["function_name"]
        assert function_name in load_function_names(), "Function name not found in available functions"
        
        # Get function information and documentation
        function_info = get_function_info(function_name)
        function_info['query'] = query
        function_info['context'] = context
        
        # Construct function parameters
        function_call = construct_function_call(function_info)
        function_call["type"] = "edit"
        return function_call

def interpret_analysis_result(query: str, function_name: str, result: any, context: str = "") -> str:
    """
    Use result_interpreter_agent to provide a human-readable interpretation of analysis results
    """
    response = client.run(
        agent=result_interpreter_agent,
        messages=[{
            "role": "user",
            "content": f"""
Query: {query}
Function: {function_name}
Result: {result}
"""
        }],
        context_variables={
            "query": query,
            "function_name": function_name,
            "result": result,
        }
    )
    return response.messages[-1]["content"]

def execute_function_call(processor: ImageProcessor, function_call_params: dict, query: str):
    """Modified to handle both analysis and editing results with interpretation"""
    try:
        # Split library and function name
        breakpoint()
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
                    if 'None' == param:
                        processed_params.append(None)
                    else:
                        processed_params.append(eval(param))
                except:
                    processed_params.append(param)
        
        if function_call_params.get("type") == "analysis":
            # Execute the function with the processed parameters
            analysis_result = func(*processed_params)
            
            # Get human-friendly interpretation
            interpretation = interpret_analysis_result(
                query=query,
                function_name=function_call_params['function_name'],
                result=analysis_result,
                context=processor.get_context_string()
            )
            
            return {
                'raw_result': analysis_result,
                'interpretation': interpretation
            }
            
        elif function_call_params.get("type") == "edit":
            # Execute the function with the processed parameters
            processed_image = func(*processed_params)
        
        # Add step and update current image
        processor.add_step(function_call_params, processed_image)
        
        return processed_image
        
    except Exception as e:
        print(f"Error executing function: {str(e)}.Use error message as more details for more detailed query for re-execution")
        return None

if __name__ == "__main__":
# Test with sample image and query
    queries = [
        # "get shape of the image",
        # "get the average pixel value of the image",
        # "get the minimum pixel value of the image",
        # "get the maximum pixel value of the image",
        # "get the number of non-zero pixels in the image",
        "get the number of pixels in the image",
        # "draw a red circle at position left corner of the image with radius 30",
        # "add gaussian blur with kernel size 11x11"
    ]

    sample_image = cv2.imread('../test.png')
    processor = ImageProcessor(sample_image)

    for query in queries:
        function_call_params = process_image_query(processor.current_image, query, processor)
        print("Query:", query)
        print("Function call params:", function_call_params)
        output = execute_function_call(processor, function_call_params, query)
        if isinstance(output, np.ndarray):
            cv2.imwrite('processed_image.png', output)
        else:
            print("ANALYSIS RESULT:", output)
        print("-" * 50)

    # Print all processing steps at the end
    print("\nProcessing Steps:")
    state = processor.get_current_state()
    for i, step in enumerate(state["steps"], 1):
        print(f"Step {i}:")
        print("Function:", json.dumps(step["function_call"], indent=2))
        # print("Image shape after step:", step["image_shape"])
        print()
import cv2
import json

def get_function_info():
    # Read function names from functions.txt
    with open('functions.txt', 'r') as f:
        function_names = [line.strip() for line in f if line.strip()]
    
    # function_info = {}
    function_names = ["get_dimensions"]
    for fn_name in function_names:
        # Remove parentheses if present
        print(fn_name)
        fn_name = fn_name.replace('()', '')
        try:
            # Get the function object from cv2
            fn = getattr(cv2, fn_name)
            print("fn: ", fn)
            # Get the help text
            help_text = str(fn.__doc__)
            # function_info[fn_name] = {
            #     "docstring": help_text,
            #     "signature": str(fn.__name__) + str(fn.__code__.co_varnames)
            # }
            return help_text

        except AttributeError:
            pass
            # Handle special cases like get_dimensions which might not be in cv2
            # function_info[fn_name] = {
            #     "docstring": "Function not found in cv2",
            #     "signature": "Unknown"
            # }
    
    # Save to JSON file
    # with open('cv_function_info.json', 'w') as f:
    #     json.dump(function_info, indent=2, sort_keys=True)
        
    # return function_info

if __name__ == "__main__":
    info = get_function_info()
    # Print first few entries as example
    print(info)
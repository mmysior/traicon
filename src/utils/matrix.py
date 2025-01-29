"""
This module provides functions to load and manipulate the contradiction matrix data.
The matrix data is stored in CSV files in the resources directory.
The module provides functions to load the matrix data and get IPs based on improving and preserving parameters.
"""
from itertools import product
from typing import List, Union, Dict
import json
import importlib.resources as pkg_resources
import numpy as np
import pandas as pd

import src.resources

# ------------------------------------------
# Contradiction Matrix
# ------------------------------------------

def load_params_data() -> List[Dict]:
    """
    Load parameters data from the JSON file.
    
    Returns:
        List[Dict]: List of parameter dictionaries containing uuid, index, parameter, and embedding
        
    Raises:
        FileNotFoundError: If parameters.json file is not found
        json.JSONDecodeError: If JSON file is invalid
    """
    json_path = pkg_resources.files(src.resources).joinpath('parameters.json')
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def get_parameter_by_index(index: int) -> str:
    """
    Get parameter name by its index.
    
    Args:
        index: Parameter index (1-39)
        
    Returns:
        str: Parameter name
        
    Raises:
        IndexError: If parameter with given index is not found
        TypeError: If index is not an integer
    """
    params_data = load_params_data()
    for param in params_data:
        if param["index"] == index:
            return param["parameter"]
    raise IndexError(f"Parameter with index {index} not found")

def load_matrix_data() -> np.ndarray:
    """
    Load the contradiction matrix data from CSV file.
    
    Returns:
        np.ndarray: Matrix values as numpy array
        
    Raises:
        FileNotFoundError: If matrix_values.csv file is not found
    """
    matrix = pd.read_csv(
        pkg_resources.files(src.resources).joinpath('matrix_values.csv'),
        delimiter=';',
        header=None
    )
    return matrix.values

def get_inventive_principles(
    improving_parameter: Union[int, List[int]],
    preserving_parameter: Union[int, List[int]]
) -> List[int]:
    """
    Get inventive principles from the contradiction matrix.
    
    Args:
        improving_parameter: Integer or list of integers representing improving parameters
        preserving_parameter: Integer or list of integers representing preserving parameters
    
    Returns:
        List[int]: List of unique principles sorted in ascending order
        
    Raises:
        TypeError: If input parameters are not of correct type
        ValueError: If parameters are invalid
    """
    # Load matrix data
    matrix_array = load_matrix_data()

    # Convert single integers to lists for consistent handling
    if not isinstance(improving_parameter, list):
        improving_parameter = [improving_parameter]
    if not isinstance(preserving_parameter, list):
        preserving_parameter = [preserving_parameter]

    # Validate parameter values
    if not all(isinstance(x, int) for x in improving_parameter + preserving_parameter):
        raise TypeError("All parameters must be integers")

    if not all(x > 0 for x in improving_parameter + preserving_parameter):
        raise ValueError("All parameters must be positive integers")

    # Convert to 0-based indexing
    row_index = [i-1 for i in improving_parameter]
    col_index = [i-1 for i in preserving_parameter]

    principles_set = set()

    for row, col in product(row_index, col_index):
        if row == col:
            continue
        string_output = matrix_array[row, col]
        if string_output is np.nan:
            continue
        list_output = string_output.split(',')
        principles_set.update(int(i.strip()) for i in list_output)

    # Convert the set to a sorted list
    principles_list = sorted(principles_set)
    return principles_list

def get_random_principles(exclude_list: List[int] | None = None, n: int = 4) -> List[int]:
    """
    Get random principles excluding specified ones.
    
    Args:
        exclude_list: List of principles to exclude, defaults to empty list
        n: Number of principles to return, defaults to 4
        
    Returns:
        List[int]: List of random principles
        
    Raises:
        ValueError: If not enough available numbers to select
        TypeError: If n is not an integer
    """
    if n < 1:
        raise ValueError("n must be positive")

    # Ensure exclude_list is a valid list
    if exclude_list is None:
        exclude_list = []

    # Create a set of numbers from 1 to 40, excluding the numbers in exclude_list
    available_numbers = set(range(1, 41)) - set(exclude_list)

    if len(available_numbers) < n:
        raise ValueError("Not enough available numbers to select the desired amount.")

    return np.random.choice(list(available_numbers), size=n, replace=False).tolist()

def get_principle_description(principle_index: int) -> str:
    """
    Get the description of a principle by its index.
    
    Args:
        principle_index: Index of the principle (1-40)
        
    Returns:
        str: Principle description
        
    Raises:
        IndexError: If principle_index is not between 1 and 40
        TypeError: If principle_index is not an integer
    """
    if not 1 <= principle_index <= 40:
        raise IndexError(f"Invalid principle index: {principle_index}. Must be between 1 and 40.")

    json_path = pkg_resources.files(src.resources).joinpath('principles.json')
    with open(json_path, encoding="utf-8") as f:
        principles = json.load(f)
    for principle in principles:
        if principle["index"] == principle_index:
            return principle["description"]
    raise IndexError(f"Principle with index {principle_index} not found")

def get_principle_name(principle_index: int) -> str:
    """
    Get the name of a principle by its index.
    
    Args:
        principle_index: Index of the principle (1-40)
        
    Returns:
        str: Principle name
        
    Raises:
        IndexError: If principle_index is not between 1 and 40
        TypeError: If principle_index is not an integer
    """
    if not 1 <= principle_index <= 40:
        raise IndexError(f"Invalid principle index: {principle_index}. Must be between 1 and 40.")

    json_path = pkg_resources.files(src.resources).joinpath('principles.json')
    with open(json_path, encoding="utf-8") as f:
        principles = json.load(f)
    for principle in principles:
        if principle["index"] == principle_index:
            return principle["name"]
    raise IndexError(f"Principle with index {principle_index} not found")

import numpy as np

def read(file_path, flag_show = False):
    """
    Reads the integral quantities from the specified file into a structured numpy array.

    Parameters:
    - file_path (str): Path to the ASCII file containing the integral quantities.

    Returns:
    - np.ndarray: A structured numpy array with named columns for each integral quantity.

    Examples:
        data = read_integral_quantities("path/to/integral_quantities.txt")
    """
    try:
        # Use numpy's genfromtxt to handle the header and read data
        # We skip the header line with column units and only read data
        # Read the header line to get the column names
        with open(file_path, 'r') as f:
                    header_line = f.readline().strip().split()
                    column_names = [header.replace('#', '').split('[')[0].strip() for header in header_line if header]
                    if flag_show:
                        print(column_names)
                    
        # Load the data, skipping all lines starting with #
        data = np.genfromtxt(file_path, names=column_names, dtype=None, encoding="utf-8")

        # Display the first few rows to confirm the data is read correctly
        if flag_show:
            print("Data successfully loaded. Here are the first few rows:")
            print(data[:5])  # Print first 5 rows for inspection
        
        return data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError:
        print("Error in reading file. Check the format and delimiter.")
    except Exception as e:
        print(f"An error occurred: {e}")
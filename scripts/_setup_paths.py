import sys
import os

def add_project_root_to_path():
    """
    Adds the project's root directory to sys.path if it's not already there.
    This allows importing modules directly from subdirectories like 'scripts'.
    Assumes this file is within the project structure (e.g., in 'scripts/').
    """
    # Determine the project root dynamically.
    # This assumes the project root is two levels up from this file,
    # or one level up from the 'scripts' directory.
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, os.pardir, os.pardir))
    # Or, if _setup_paths.py is directly in 'scripts' and 'scripts' is at the root:
    # project_root = os.path.abspath(os.path.join(current_file_dir, os.pardir)) # This would be correct if _setup_paths.py is directly in the project root.

    # Let's adjust this for the specific case:
    # _setup_paths.py is in MULTI-POINT-STATISTICS/scripts/
    # The project root is MULTI-POINT-STATISTICS/

    # So, from current_file_dir (scripts/), go up one level to get to the root.
    actual_project_root = os.path.abspath(os.path.join(current_file_dir, os.pardir))


    if actual_project_root not in sys.path:
        sys.path.insert(0, actual_project_root)
        print(f"Added project root to sys.path: {actual_project_root}")
    # else:
    #     print(f"Project root already in sys.path: {actual_project_root}")

if __name__ == "__main__":
    # This block allows you to test the function independently
    add_project_root_to_path()
    print("Current sys.path after running _setup_paths.py:")
    for p in sys.path:
        print(f"  {p}")

    # You could also add a dummy import test here
    try:
        from scripts.utils import helpfunc
        print("\nTest import successful: from scripts.utils import helpfunc")
    except ImportError as e:
        print(f"\nTest import failed: {e}. 'scripts' might not be found.")
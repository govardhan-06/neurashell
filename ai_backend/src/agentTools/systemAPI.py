import ctypes
import os
import win32file
import win32con
import glob
from typing import Annotated, List
from langchain_core.tools import tool

@tool
def search_files(directory: Annotated[str, "The directory to search in"], 
                 extension: Annotated[str, "The file extension to search for"]) -> List[str]:
    """Search for files with a specific extension in a directory (Windows API)."""
    search_pattern = os.path.join(directory, f"*{extension}")
    print(search_pattern)
    print(glob.glob(search_pattern))
    return glob.glob(search_pattern)

@tool
def read_file(file_path: Annotated[str, "The path to the file to read"]) -> str:
    """Read a file's content using Windows API."""
    handle = win32file.CreateFile(
        file_path,
        win32con.GENERIC_READ,
        0,  # No sharing
        None,
        win32con.OPEN_EXISTING,
        0,
        None
    )
    _, data = win32file.ReadFile(handle, 1024 * 1024)  # Read 1MB chunks
    win32file.CloseHandle(handle)
    return data.decode("utf-8", errors="ignore")

@tool
def write_file(file_path: Annotated[str, "The path to the file to write"],
               content: Annotated[str, "The content to write to the file"]):
    """Write content to a file using Windows API."""
    handle = win32file.CreateFile(
        file_path,
        win32con.GENERIC_WRITE,
        0,
        None,
        win32con.CREATE_ALWAYS,
        0,
        None
    )
    win32file.WriteFile(handle, content.encode("utf-8"))
    win32file.CloseHandle(handle)

@tool
def move_file(src: Annotated[str, "The source file path"],
              dest: Annotated[str, "The destination file path"]):
    """Move a file using Windows API."""
    ctypes.windll.kernel32.MoveFileW(ctypes.c_wchar_p(src), ctypes.c_wchar_p(dest))

@tool
def delete_file(file_path: Annotated[str, "The path to the file to delete"]):
    """Delete a file using Windows API."""
    win32file.DeleteFile(file_path)

#Testing the system API
def test_system_api():
    # Create a test directory
    test_dir = os.path.join(os.path.dirname(__file__), "test_files")
    os.makedirs(test_dir, exist_ok=True)

    # Test write_file
    test_file = os.path.join(test_dir, "test.txt")
    test_content = "Hello, this is a test file!"
    write_file(test_file, test_content)
    print(f"Created file: {test_file}")

    # Test read_file
    content = read_file(test_file)
    print(f"Read content: {content}")
    assert content == test_content

    # Test search_files
    files = search_files(test_dir, "txt")
    print(f"Found files: {files}")
    assert len(files) > 0

    # Test move_file
    moved_file = os.path.join(test_dir, "moved.txt")
    move_file(test_file, moved_file)
    print(f"Moved file to: {moved_file}")
    assert os.path.exists(moved_file)

    # Test delete_file
    delete_file(moved_file)
    print("Deleted file")
    assert not os.path.exists(moved_file)

    # Clean up
    os.rmdir(test_dir)
    print("Test directory removed")

if __name__ == "__main__":
    test_system_api()
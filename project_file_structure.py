import os
from pathlib import Path

def create_folder_structure(root_path, prefix="", ignore_folders=None, skip_contents_folders=None):
    """
    Create a visual representation of folder structure.
    
    Args:
        root_path: Path to scan
        prefix: Prefix for tree visualization
        ignore_folders: Set of folder names to ignore completely
        skip_contents_folders: Set of folder paths to show but skip their contents
    
    Returns:
        List of strings representing the folder structure
    """
    if ignore_folders is None:
        ignore_folders = set()
    if skip_contents_folders is None:
        skip_contents_folders = set()
    
    structure = []
    
    try:
        # Get all items in the directory
        items = list(Path(root_path).iterdir())
        # Sort items: directories first, then files
        items.sort(key=lambda x: (x.is_file(), x.name.lower()))
        
        for i, item in enumerate(items):
            # Skip ignored folders
            if item.name in ignore_folders:
                continue
                
            # Determine if this is the last item (excluding ignored items)
            remaining_items = [x for x in items[i:] if x.name not in ignore_folders]
            is_last = len(remaining_items) == 1
            
            # Create the appropriate prefix
            current_prefix = "└── " if is_last else "├── "
            structure.append(f"{prefix}{current_prefix}{item.name}")
            
            # If it's a directory, check if we should recurse into it
            if item.is_dir():
                # Check if this folder's contents should be skipped
                # Create a normalized path for comparison
                try:
                    relative_path = str(item.relative_to(Path.cwd())).replace('\\', '/')
                except ValueError:
                    # If relative_to fails, use the item path as is
                    relative_path = str(item).replace('\\', '/')
                
                # Also check against the item name and full path
                item_path_str = str(item).replace('\\', '/')
                should_skip = (relative_path in skip_contents_folders or 
                              item_path_str in skip_contents_folders or
                              item.name in skip_contents_folders)
                
                if should_skip:
                    # Show the folder but skip its contents
                    extension_prefix = "    " if is_last else "│   "
                    structure.append(f"{prefix}{extension_prefix}[Contents skipped]")
                else:
                    # Recursively get its structure
                    extension_prefix = "    " if is_last else "│   "
                    sub_structure = create_folder_structure(
                        item, 
                        prefix + extension_prefix, 
                        ignore_folders,
                        skip_contents_folders
                    )
                    structure.extend(sub_structure)
                
    except PermissionError:
        structure.append(f"{prefix}[Permission Denied]")
    except Exception as e:
        structure.append(f"{prefix}[Error: {str(e)}]")
    
    return structure

def scan_folders(target_folders, ignore_folders=None, skip_contents_folders=None, output_file="folder_structure.txt"):
    """
    Scan specified folders and save structure to file.
    
    Args:
        target_folders: List of folder names to scan
        ignore_folders: Set of folder names to ignore completely
        skip_contents_folders: Set of folder paths to show but skip their contents
        output_file: Output file name
    """
    if ignore_folders is None:
        ignore_folders = set()
    if skip_contents_folders is None:
        skip_contents_folders = set()
    
    all_structure = []
    
    for folder in target_folders:
        folder_path = Path(folder)
        
        if folder_path.exists() and folder_path.is_dir():
            all_structure.append(f"\n{'='*50}")
            all_structure.append(f"FOLDER STRUCTURE: {folder}")
            all_structure.append(f"{'='*50}")
            all_structure.append(f"{folder}/")
            
            structure = create_folder_structure(
                folder_path, 
                "", 
                ignore_folders, 
                skip_contents_folders
            )
            all_structure.extend(structure)
            all_structure.append("")  # Empty line for spacing
            
        else:
            all_structure.append(f"\n[WARNING] Folder '{folder}' not found or not accessible")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FOLDER STRUCTURE REPORT\n")
        f.write(f"Generated on: {Path.cwd()}\n")
        f.write(f"Ignored folders: {', '.join(ignore_folders) if ignore_folders else 'None'}\n")
        f.write(f"Skip contents folders: {', '.join(skip_contents_folders) if skip_contents_folders else 'None'}\n")
        f.write("\n".join(all_structure))
    
    print(f"Folder structure saved to: {output_file}")
    
    # Also print to console
    print("\n".join(all_structure))

if __name__ == "__main__":
    # Define folders to scan
    target_folders = ["docker", "src", "telegram_app_handler"]
    
    # Define folders to ignore completely
    ignore_folders = {".github", ".pytest_cache", ".venv", ".vscode"}
    
    # Define folders to show but skip their contents
    skip_contents_folders = {"docker/influxdb/data"}
    
    # Scan and save
    scan_folders(target_folders, ignore_folders, skip_contents_folders)
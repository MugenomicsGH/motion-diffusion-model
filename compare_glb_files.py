import json
import os
import sys

def load_json(file_path):
    """Load JSON data from a file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_animations(example_data, our_data):
    """Compare animation data between two files"""
    print("\n=== ANIMATION COMPARISON ===")
    
    example_animations = example_data.get("animations", [])
    our_animations = our_data.get("animations", [])
    
    print(f"Example animations: {len(example_animations)}")
    print(f"Our animations: {len(our_animations)}")
    
    if example_animations and our_animations:
        example_anim = example_animations[0]
        our_anim = our_animations[0]
        
        print(f"\nExample animation name: {example_anim.get('name')}")
        print(f"Our animation name: {our_anim.get('name')}")
        
        print(f"\nExample channels: {example_anim.get('channels')}")
        print(f"Our channels: {our_anim.get('channels')}")
        
        # Check animation paths
        example_paths = {}
        for channel in example_anim.get("channelsList", []):
            path = channel.get("target", {}).get("path")
            example_paths[path] = example_paths.get(path, 0) + 1
            
        our_paths = {}
        for channel in our_anim.get("channelsList", []):
            path = channel.get("target", {}).get("path")
            our_paths[path] = our_paths.get(path, 0) + 1
        
        print("\nAnimation paths:")
        print(f"Example: {example_paths}")
        print(f"Ours: {our_paths}")
        
        # Check interpolation methods
        example_interp = {}
        for sampler in example_anim.get("samplersList", []):
            interp = sampler.get("interpolation")
            example_interp[interp] = example_interp.get(interp, 0) + 1
            
        our_interp = {}
        for sampler in our_anim.get("samplersList", []):
            interp = sampler.get("interpolation")
            our_interp[interp] = our_interp.get(interp, 0) + 1
        
        print("\nInterpolation methods:")
        print(f"Example: {example_interp}")
        print(f"Ours: {our_interp}")

def compare_nodes(example_data, our_data):
    """Compare node structure between two files"""
    print("\n=== NODE COMPARISON ===")
    
    example_nodes = example_data.get("nodes", [])
    our_nodes = our_data.get("nodes", [])
    
    print(f"Example nodes: {len(example_nodes)}")
    print(f"Our nodes: {len(our_nodes)}")
    
    # Compare node names
    example_names = [node.get("name") for node in example_data.get("nodesList", [])]
    our_names = [node.get("name") for node in our_data.get("nodesList", [])]
    
    print("\nExample node name prefixes:")
    prefixes = {}
    for name in example_names:
        if name:
            prefix = name.split(':')[0] if ':' in name else name
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}: {count}")
    
    print("\nOur node names:")
    for name in our_names:
        print(f"  {name}")

def compare_coordinate_systems(example_data, our_data):
    """Try to determine coordinate system differences"""
    print("\n=== COORDINATE SYSTEM ANALYSIS ===")
    
    example_nodes = example_data.get("nodesList", [])
    our_nodes = our_data.get("nodesList", [])
    
    # Check transformations for root nodes
    print("Example root transformations:")
    for node in example_nodes[:5]:  # Just check the first few nodes
        name = node.get("name", "")
        if "Hips" in name or "root" in name.lower() or "Armature" in name:
            print(f"  {name}:")
            print(f"    Translation: {node.get('translation')}")
            print(f"    Rotation: {node.get('rotation')}")
            print(f"    Scale: {node.get('scale')}")
    
    print("\nOur root transformations:")
    for node in our_nodes[:5]:  # Just check the first few nodes
        name = node.get("name", "")
        if "hips" in name.lower() or "root" in name.lower() or "Armature" in name:
            print(f"  {name}:")
            print(f"    Translation: {node.get('translation')}")
            print(f"    Rotation: {node.get('rotation')}")
            print(f"    Scale: {node.get('scale')}")

    # Check animation data for translation/rotation patterns
    example_animations = example_data.get("animations", [])
    our_animations = our_data.get("animations", [])
    
    if example_animations and "channelsList" in example_animations[0]:
        example_channels = example_animations[0]["channelsList"]
        # Find a translation channel
        for channel in example_channels:
            if channel.get("target", {}).get("path") == "translation":
                print("\nExample translation accessors:")
                accessor_idx = channel.get("sampler", {}).get("output")
                if accessor_idx is not None and "accessorsList" in example_data:
                    accessor = example_data["accessorsList"][accessor_idx]
                    print(f"  Min: {accessor.get('min')}")
                    print(f"  Max: {accessor.get('max')}")
                break
    
    if our_animations and "channelsList" in our_animations[0]:
        our_channels = our_animations[0]["channelsList"]
        # Find a translation channel
        for channel in our_channels:
            if channel.get("target", {}).get("path") == "translation":
                print("\nOur translation accessors:")
                accessor_idx = channel.get("sampler", {}).get("output")
                if accessor_idx is not None and "accessorsList" in our_data:
                    accessor = our_data["accessorsList"][accessor_idx]
                    print(f"  Min: {accessor.get('min')}")
                    print(f"  Max: {accessor.get('max')}")
                break

def main():
    example_file = "female-walk-inspect.json"
    our_file = "our-walk-inspect.json"
    
    if not os.path.exists(example_file):
        print(f"Error: {example_file} not found")
        return
    
    if not os.path.exists(our_file):
        print(f"Error: {our_file} not found")
        return
    
    example_data = load_json(example_file)
    our_data = load_json(our_file)
    
    if not example_data or not our_data:
        return
    
    print(f"Comparing {example_file} and {our_file}")
    
    # Basic info
    print("\n=== BASIC INFO ===")
    print(f"Example generator: {example_data.get('asset', {}).get('generator')}")
    print(f"Our generator: {our_data.get('asset', {}).get('generator')}")
    
    print(f"\nExample file size: {example_data.get('size', 'unknown')}")
    print(f"Our file size: {our_data.get('size', 'unknown')}")
    
    # Compare animations
    compare_animations(example_data, our_data)
    
    # Compare nodes
    compare_nodes(example_data, our_data)
    
    # Analyze coordinate systems
    compare_coordinate_systems(example_data, our_data)

if __name__ == "__main__":
    main() 
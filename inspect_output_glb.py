import os
import sys
import pygltflib
import json

# Path to the GLB file to inspect
glb_path = 'outputs/glb/motion_20250319_204823.glb'

def inspect_glb(file_path):
    """Inspect a GLB file and print detailed information about its structure"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Inspecting GLB file: {file_path} ({os.path.getsize(file_path)} bytes)")
    
    try:
        # Load the GLB file
        gltf = pygltflib.GLTF2().load(file_path)
        
        # Print basic information
        print("\n=== GENERAL INFO ===")
        print(f"Asset version: {gltf.asset.version}")
        print(f"Asset generator: {gltf.asset.generator}")
        
        # Scenes
        print("\n=== SCENES ===")
        print(f"Number of scenes: {len(gltf.scenes)}")
        print(f"Default scene: {gltf.scene}")
        
        # Nodes
        print("\n=== NODES ===")
        print(f"Number of nodes: {len(gltf.nodes)}")
        
        # Print node hierarchy
        if gltf.scenes:
            print("\nNode hierarchy:")
            
            def print_hierarchy(node_idx, depth=0):
                try:
                    node = gltf.nodes[node_idx]
                    name = node.name or f"Node {node_idx}"
                    print("  " * depth + f"- {name}")
                    
                    # Check for rotations and translations
                    if hasattr(node, 'rotation'):
                        if node.rotation != [0, 0, 0, 1] and node.rotation is not None:
                            print("  " * depth + f"  rotation: {node.rotation}")
                    
                    if hasattr(node, 'translation'):
                        if node.translation != [0, 0, 0] and node.translation is not None:
                            print("  " * depth + f"  translation: {node.translation}")
                    
                    if hasattr(node, 'children') and node.children:
                        for child_idx in node.children:
                            print_hierarchy(child_idx, depth + 1)
                except Exception as e:
                    print("  " * depth + f"- Error accessing node {node_idx}: {str(e)}")
            
            # Print hierarchy for all root nodes in the first scene
            scene = gltf.scenes[gltf.scene]
            if hasattr(scene, 'nodes') and scene.nodes:
                for root_node_idx in scene.nodes:
                    print_hierarchy(root_node_idx)
        
        # Animations
        print("\n=== ANIMATIONS ===")
        print(f"Number of animations: {len(gltf.animations)}")
        
        for i, animation in enumerate(gltf.animations):
            print(f"\nAnimation {i}:")
            print(f"  Name: {animation.name}")
            print(f"  Channels: {len(animation.channels)}")
            print(f"  Samplers: {len(animation.samplers)}")
            
            # Print sample of channels
            if animation.channels:
                print("  Sample channels:")
                for j, channel in enumerate(animation.channels[:min(5, len(animation.channels))]):
                    target = channel.target
                    node_id = target.node if hasattr(target, 'node') else None
                    node_name = gltf.nodes[node_id].name if node_id is not None and node_id < len(gltf.nodes) else "Unknown"
                    print(f"    Channel {j}: Target node {node_id} ({node_name}), Path: {target.path}")
                    
                    # Look at the sampler
                    if hasattr(channel, 'sampler') and channel.sampler < len(animation.samplers):
                        sampler = animation.samplers[channel.sampler]
                        input_acc = gltf.accessors[sampler.input] if sampler.input < len(gltf.accessors) else None
                        output_acc = gltf.accessors[sampler.output] if sampler.output < len(gltf.accessors) else None
                        
                        print(f"      Sampler interpolation: {sampler.interpolation}")
                        if input_acc:
                            print(f"      Input accessor: {sampler.input}, Count: {input_acc.count}, Type: {input_acc.type}")
                        if output_acc:
                            print(f"      Output accessor: {sampler.output}, Count: {output_acc.count}, Type: {output_acc.type}")
        
        # Buffers and buffer views
        print("\n=== BUFFERS AND VIEWS ===")
        print(f"Number of buffers: {len(gltf.buffers)}")
        print(f"Number of buffer views: {len(gltf.bufferViews)}")
        print(f"Number of accessors: {len(gltf.accessors)}")
        
        # Check buffer content
        for i, buffer in enumerate(gltf.buffers):
            print(f"\nBuffer {i}:")
            print(f"  Byte length: {buffer.byteLength}")
            print(f"  URI: {buffer.uri if hasattr(buffer, 'uri') else 'None'}")
            
            # Check binary blob
            binary_blob = gltf.binary_blob()
            binary_length = len(binary_blob) if binary_blob else 0
            print(f"  Binary blob length: {binary_length}")
            
            if binary_blob and binary_length > 0:
                print(f"  Binary blob first 20 bytes: {binary_blob[:min(20, binary_length)]}")
        
        return gltf
        
    except Exception as e:
        import traceback
        print(f"Error inspecting GLB file: {str(e)}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default
    file_path = sys.argv[1] if len(sys.argv) > 1 else glb_path
    inspect_glb(file_path) 
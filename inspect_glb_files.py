import os
import sys
from pygltflib import GLTF2

def inspect_glb(glb_path):
    """Inspect a GLB file and print its structure"""
    print(f"\n=== INSPECTING: {glb_path} ===")
    
    # Get file size
    file_size = os.path.getsize(glb_path)
    print(f"File size: {file_size} bytes")
    
    # Load the GLB file
    gltf = GLTF2.load(glb_path)
    
    # General info
    print("\n== General Info ==")
    print(f"Asset version: {gltf.asset.version}")
    print(f"Asset generator: {gltf.asset.generator}")
    
    # Nodes
    print("\n== Nodes ==")
    print(f"Number of nodes: {len(gltf.nodes)}")
    print("Top-level nodes:")
    for i, node in enumerate(gltf.nodes[:5]):
        print(f"  Node {i}: name={node.name}, children={node.children}")
        print(f"    translation={node.translation}, rotation={node.rotation}, scale={node.scale}")
    
    # Animations
    print("\n== Animations ==")
    print(f"Number of animations: {len(gltf.animations)}")
    for i, animation in enumerate(gltf.animations):
        print(f"\nAnimation {i}:")
        print(f"  Name: {animation.name}")
        print(f"  Channels: {len(animation.channels)}")
        print(f"  Samplers: {len(animation.samplers)}")
        
        # Count animation channel paths
        paths = {}
        for channel in animation.channels:
            path = channel.target.path
            paths[path] = paths.get(path, 0) + 1
        print(f"  Paths: {paths}")
        
        # Check interpolation methods
        interpolations = {}
        for sampler in animation.samplers:
            interp = sampler.interpolation
            interpolations[interp] = interpolations.get(interp, 0) + 1
        print(f"  Interpolations: {interpolations}")
        
        # Sample a few channels
        print("\n  Sample channels:")
        for j, channel in enumerate(animation.channels[:5]):
            target_node_idx = channel.target.node
            target_node_name = gltf.nodes[target_node_idx].name if target_node_idx < len(gltf.nodes) else "unknown"
            sampler = animation.samplers[channel.sampler]
            
            input_accessor = gltf.accessors[sampler.input]
            output_accessor = gltf.accessors[sampler.output]
            
            print(f"    Channel {j}:")
            print(f"      Target: node {target_node_idx} ({target_node_name}) - {channel.target.path}")
            print(f"      Sampler interpolation: {sampler.interpolation}")
            print(f"      Input accessor: type={input_accessor.type}, count={input_accessor.count}")
            print(f"      Output accessor: type={output_accessor.type}, count={output_accessor.count}")
    
    # Buffers
    print("\n== Buffers ==")
    print(f"Number of buffers: {len(gltf.buffers)}")
    print(f"Number of buffer views: {len(gltf.bufferViews)}")
    print(f"Number of accessors: {len(gltf.accessors)}")
    
    for i, buffer in enumerate(gltf.buffers):
        print(f"\nBuffer {i}:")
        print(f"  Byte length: {buffer.byteLength}")
        
        binary_blob = gltf._binary_data if hasattr(gltf, '_binary_data') else None
        binary_len = len(binary_blob) if binary_blob else 0
        print(f"  Binary blob length: {binary_len}")
        
        if binary_blob and binary_len > 0:
            print(f"  First few bytes: {binary_blob[:20]}")

def main():
    example_glb = "api/female-walk.glb"
    our_glb = "outputs/glb/motion_20250319_210109.glb"
    
    # Inspect the example GLB
    if os.path.exists(example_glb):
        inspect_glb(example_glb)
    else:
        print(f"Error: {example_glb} not found")
    
    # Inspect our GLB
    if os.path.exists(our_glb):
        inspect_glb(our_glb)
    else:
        print(f"Error: {our_glb} not found")

if __name__ == "__main__":
    main() 
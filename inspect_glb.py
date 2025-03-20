import pygltflib
import json

# Load the GLB file
gltf = pygltflib.GLTF2().load('api/female-walk.glb')

# Print basic information
print("Number of nodes:", len(gltf.nodes))
print("Number of animations:", len(gltf.animations))

if gltf.animations:
    anim = gltf.animations[0]
    print("\nAnimation name:", anim.name)
    print("Number of channels:", len(anim.channels))
    print("Number of samplers:", len(anim.samplers))
    
    # Print details of first few channels
    print("\nFirst 5 animation channels:")
    for i, channel in enumerate(anim.channels[:5]):
        target = channel.target
        target_node = gltf.nodes[target.node] if hasattr(target, 'node') else "N/A"
        node_name = target_node.name if hasattr(target_node, 'name') else "Unnamed"
        print(f"Channel {i}:")
        print(f"  Target node: {target.node} ({node_name})")
        print(f"  Target path: {target.path}")
        print(f"  Sampler: {channel.sampler}")
        
        # Print sampler details
        sampler = anim.samplers[channel.sampler]
        print(f"  Interpolation: {sampler.interpolation}")
        print(f"  Input accessor: {sampler.input}")
        print(f"  Output accessor: {sampler.output}")
        
    # Print node hierarchy
    print("\nNode hierarchy:")
    
    def print_hierarchy(node_idx, depth=0):
        node = gltf.nodes[node_idx]
        name = node.name or f"Node {node_idx}"
        print("  " * depth + f"- {name}")
        
        if hasattr(node, 'children') and node.children:
            for child_idx in node.children:
                print_hierarchy(child_idx, depth + 1)
    
    # Assuming scene 0 is the main scene
    if gltf.scenes and gltf.scenes[0].nodes:
        for root_node_idx in gltf.scenes[0].nodes:
            print_hierarchy(root_node_idx) 
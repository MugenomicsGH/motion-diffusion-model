import os
import logging
import numpy as np
import struct
import math
from pygltflib import (
    GLTF2, Scene, Node, Buffer, BufferView, Accessor, Animation,
    AnimationChannel, AnimationChannelTarget, AnimationSampler,
    Asset
)
# Direct constants instead of enums
from pygltflib import (
    SCALAR, VEC3, VEC4,  # AccessorType values
    FLOAT,  # ComponentType value
    ARRAY_BUFFER  # BufferTarget value
)
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

logger = logging.getLogger(__name__)

# Mapping of VRM bone names to MDM joint names
VRM_BONE_MAPPING = {
    'hips': 'root',
    'spine': 'spine',
    'chest': 'chest',
    'neck': 'neck',
    'head': 'head',
    'leftShoulder': 'shoulder.L',
    'leftUpperArm': 'upper_arm.L',
    'leftLowerArm': 'lower_arm.L',
    'leftHand': 'hand.L',
    'rightShoulder': 'shoulder.R',
    'rightUpperArm': 'upper_arm.R',
    'rightLowerArm': 'lower_arm.R',
    'rightHand': 'hand.R',
    'leftUpperLeg': 'upper_leg.L',
    'leftLowerLeg': 'lower_leg.L',
    'leftFoot': 'foot.L',
    'rightUpperLeg': 'upper_leg.R',
    'rightLowerLeg': 'lower_leg.R',
    'rightFoot': 'foot.R'
}

def normalize(v):
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v

def look_at_quaternion(forward, up=np.array([0, 1, 0])):
    """
    Create a quaternion that orients an object to look in the specified direction
    
    Args:
        forward: The forward direction vector
        up: The up direction vector (default: world up)
        
    Returns:
        A quaternion [x, y, z, w] representing the rotation
    """
    forward = normalize(forward)
    
    # Check if forward is parallel to up
    if np.abs(np.dot(forward, up)) > 0.999:
        # Choose a different up vector
        up = np.array([0, 0, 1])
        if np.abs(np.dot(forward, up)) > 0.999:
            up = np.array([1, 0, 0])
    
    right = normalize(np.cross(forward, up))
    up = normalize(np.cross(right, forward))
    
    # Create rotation matrix
    m = np.zeros((3, 3))
    m[0, 0] = right[0]
    m[0, 1] = up[0]
    m[0, 2] = -forward[0]
    m[1, 0] = right[1]
    m[1, 1] = up[1]
    m[1, 2] = -forward[1]
    m[2, 0] = right[2]
    m[2, 1] = up[2]
    m[2, 2] = -forward[2]
    
    # Convert rotation matrix to quaternion
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])

def motion_to_glb(motion_data, output_path, fps=30):
    """
    Convert motion data to GLB format with VRM-compatible skeleton and generate MP4 visualization
    
    Args:
        motion_data (np.ndarray): Motion data array from MDM with shape (batch_size, n_joints, 3, n_frames)
        output_path (str): Path to save the GLB file
        fps (int): Target frames per second for the animation (note: input motion is always at 20 fps)
    """
    try:
        # Validate motion data
        if motion_data is None:
            raise ValueError("Motion data cannot be None")
            
        if not isinstance(motion_data, np.ndarray):
            logger.info("Converting motion data to numpy array")
            motion_data = np.array(motion_data)
            
        if len(motion_data.shape) != 4:
            raise ValueError(f"Expected 4D motion data array (batch_size, n_joints, 3, n_frames), got shape {motion_data.shape}")
            
        batch_size, n_joints, n_dims, n_frames = motion_data.shape
        logger.info(f"Motion data shape: batch_size={batch_size}, n_joints={n_joints}, n_dims={n_dims}, n_frames={n_frames}")
        
        # Calculate actual duration in seconds (motion data is always at 20 fps)
        duration_seconds = n_frames / 20.0
        logger.info(f"Motion duration: {duration_seconds:.2f} seconds at 20 fps input")
        
        if n_dims != 3:
            raise ValueError(f"Expected 3 dimensions per joint, got {n_dims}")
            
        # Take first batch and transpose to (n_frames, n_joints, 3)
        logger.info("Reshaping motion data")
        motion_data = motion_data[0].transpose(2, 0, 1)
        logger.info(f"Reshaped motion data shape: {motion_data.shape}")

        # Debug: Analyze motion ranges
        logger.info("=== Motion Data Analysis ===")
        joint_positions = {}
        for joint_idx, mdm_joint in enumerate(VRM_BONE_MAPPING.values()):
            data = motion_data[:, joint_idx]
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)
            range_vals = max_vals - min_vals
            
            # Store positions for later use
            joint_positions[mdm_joint] = {
                'data': data,
                'mean': mean_vals,
                'min': min_vals,
                'max': max_vals,
                'range': range_vals
            }
            
            logger.info(f"\nJoint: {mdm_joint}")
            logger.info(f"X - min: {min_vals[0]:.4f}, max: {max_vals[0]:.4f}, mean: {mean_vals[0]:.4f}, std: {std_vals[0]:.4f}, range: {range_vals[0]:.4f}")
            logger.info(f"Y - min: {min_vals[1]:.4f}, max: {max_vals[1]:.4f}, mean: {mean_vals[1]:.4f}, std: {std_vals[1]:.4f}, range: {range_vals[1]:.4f}")
            logger.info(f"Z - min: {min_vals[2]:.4f}, max: {max_vals[2]:.4f}, mean: {mean_vals[2]:.4f}, std: {std_vals[2]:.4f}, range: {range_vals[2]:.4f}")
            
            # Print first few frames for root joint to understand data pattern
            if mdm_joint == 'root':
                logger.info("\nFirst 5 frames of root joint motion:")
                for frame in range(min(5, len(data))):
                    logger.info(f"Frame {frame}: X={data[frame][0]:.4f}, Y={data[frame][1]:.4f}, Z={data[frame][2]:.4f}")
        logger.info("=== End Motion Data Analysis ===\n")

        # Create GLTF structure
        logger.info("Creating GLTF structure")
        gltf = GLTF2()
        gltf.scene = 0
        gltf.scenes = [Scene(nodes=[0])]  # Root node
        gltf.asset = Asset(version="2.0", generator="MDM-to-GLB Converter")

        # Initialize empty lists and binary data
        gltf.nodes = []
        gltf.meshes = []
        gltf.animations = []
        gltf.bufferViews = []
        gltf.accessors = []
        gltf.buffers = []
        binary_data = bytearray()

        # Create skeleton nodes with proper transforms
        logger.info("Creating skeleton nodes")
        joint_nodes = {}
        
        # Create Armature node first
        armature_node = Node(
            name="Armature",
            translation=[0, 0, 0],
            rotation=[0, 0, 0, 1],
            scale=[1, 1, 1]
        )
        gltf.nodes.append(armature_node)
        root_node_idx = 0
        
        # Create all bone nodes with identity transforms
        for joint_name in VRM_BONE_MAPPING.keys():
            node = Node(
                name=joint_name,
                translation=[0, 0, 0],
                rotation=[0, 0, 0, 1],  # Identity quaternion
                scale=[1, 1, 1]
            )
            gltf.nodes.append(node)
            joint_nodes[joint_name] = len(gltf.nodes) - 1

        # Set up parent-child relationships
        logger.info("Setting up skeleton hierarchy")
        
        # Create the hierarchy mapping
        hierarchy = {
            'hips': ['spine', 'leftUpperLeg', 'rightUpperLeg'],
            'spine': ['chest'],
            'chest': ['neck', 'leftShoulder', 'rightShoulder'],
            'neck': ['head'],
            'leftShoulder': ['leftUpperArm'],
            'leftUpperArm': ['leftLowerArm'],
            'leftLowerArm': ['leftHand'],
            'rightShoulder': ['rightUpperArm'],
            'rightUpperArm': ['rightLowerArm'],
            'rightLowerArm': ['rightHand'],
            'leftUpperLeg': ['leftLowerLeg'],
            'leftLowerLeg': ['leftFoot'],
            'rightUpperLeg': ['rightLowerLeg'],
            'rightLowerLeg': ['rightFoot']
        }
        
        # First set hips as child of Armature
        armature_node.children = [joint_nodes['hips']]
        
        # Then set up the rest of the hierarchy
        for parent, children in hierarchy.items():
            parent_idx = joint_nodes[parent]
            child_indices = [joint_nodes[child] for child in children if child in joint_nodes]
            if child_indices:
                gltf.nodes[parent_idx].children = child_indices

        # Create animation
        logger.info(f"Creating animation timeline with {n_frames} frames at {fps} FPS (input motion at 20 FPS)")
        animation = Animation(
            name="motion",
            channels=[],
            samplers=[]
        )

        # Create time data for animations - adjust timing to maintain original duration
        times = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        time_data = times.tobytes()
        
        # Add time data to binary blob and create buffer view
        time_buffer_view = BufferView(
            buffer=0,
            byteOffset=len(binary_data),
            byteLength=len(time_data),
            target=ARRAY_BUFFER
        )
        binary_data.extend(time_data)
        gltf.bufferViews.append(time_buffer_view)
        time_buffer_view_idx = len(gltf.bufferViews) - 1
        
        # Create time accessor
        time_accessor = Accessor(
            bufferView=time_buffer_view_idx,
            componentType=FLOAT,
            count=n_frames,
            type=SCALAR,
            min=[float(times.min())],
            max=[float(times.max())]
        )
        gltf.accessors.append(time_accessor)
        time_accessor_idx = len(gltf.accessors) - 1
        
        # Define rest pose transformations
        logger.info("Calculating initial pose and bone transforms")
        
        # Determine the average bone vectors between joints in the reference pose (first frame)
        reference_frame = 0
        bone_vectors = {}
        bone_lengths = {}
        
        # Calculate the average position for each joint across all frames
        avg_positions = {}
        for joint_name, mdm_joint in VRM_BONE_MAPPING.items():
            if mdm_joint in joint_positions:
                avg_positions[joint_name] = joint_positions[mdm_joint]['mean']
        
        # Calculate bone vectors for the hierarchy
        for parent, children in hierarchy.items():
            parent_pos = avg_positions.get(parent)
            if parent_pos is None:
                continue
                
            for child in children:
                child_pos = avg_positions.get(child)
                if child_pos is None:
                    continue
                    
                # Calculate bone vector from parent to child
                bone_vector = child_pos - parent_pos
                bone_length = np.linalg.norm(bone_vector)
                
                bone_vectors[(parent, child)] = bone_vector
                bone_lengths[(parent, child)] = bone_length
                
                logger.info(f"Bone {parent}->{child}: vector={bone_vector}, length={bone_length:.4f}")
        
        # Process joint animations
        logger.info("Processing joint animations")
        
        # Process each joint for animations
        for vrm_joint, mdm_joint in VRM_BONE_MAPPING.items():
            try:
                if mdm_joint not in joint_positions:
                    logger.warning(f"Skipping joint {vrm_joint} (MDM joint: {mdm_joint}) - not found in positions")
                    continue
                    
                logger.info(f"Processing joint: {vrm_joint} (MDM joint: {mdm_joint})")
                joint_data = joint_positions[mdm_joint]['data']
                
                # Special case for root/hips - use position directly
                if vrm_joint == 'hips':
                    # Create position data for the root joint
                    position_data = np.zeros((n_frames, 3), dtype=np.float32)
                    
                    # Scale factor to make movements more visible
                    scale_factor = 5.0
                    
                    # Convert positions to VRM coordinate system
                    pos_min = [float('inf')] * 3
                    pos_max = [float('-inf')] * 3
                    
                    for frame in range(n_frames):
                        x, y, z = joint_data[frame]
                        
                        # Apply scale and convert to VRM coordinate system
                        pos_x = float(x * scale_factor)
                        pos_y = float(y * scale_factor)
                        pos_z = float(-z * scale_factor)  # Negate Z for VRM
                        
                        position_data[frame] = [pos_x, pos_y, pos_z]
                        
                        # Update min/max
                        pos_min[0] = min(pos_min[0], pos_x)
                        pos_min[1] = min(pos_min[1], pos_y)
                        pos_min[2] = min(pos_min[2], pos_z)
                        
                        pos_max[0] = max(pos_max[0], pos_x)
                        pos_max[1] = max(pos_max[1], pos_y)
                        pos_max[2] = max(pos_max[2], pos_z)
                    
                    # Convert position data to bytes
                    position_bytes = position_data.tobytes()
                    
                    # Create buffer view for position data
                    pos_buffer_view = BufferView(
                        buffer=0,
                        byteOffset=len(binary_data),
                        byteLength=len(position_bytes),
                        target=ARRAY_BUFFER
                    )
                    gltf.bufferViews.append(pos_buffer_view)
                    position_buffer_view_idx = len(gltf.bufferViews) - 1
                    
                    # Add position data to binary blob
                    binary_data.extend(position_bytes)
                    
                    # Create accessor for position data
                    position_accessor = Accessor(
                        bufferView=position_buffer_view_idx,
                        componentType=FLOAT,
                        count=n_frames,
                        type=VEC3,
                        min=pos_min,
                        max=pos_max
                    )
                    gltf.accessors.append(position_accessor)
                    position_accessor_idx = len(gltf.accessors) - 1
                    
                    # Create sampler for the position animation
                    pos_sampler = AnimationSampler(
                        input=time_accessor_idx,
                        output=position_accessor_idx,
                        interpolation="LINEAR"
                    )
                    animation.samplers.append(pos_sampler)
                    position_sampler_idx = len(animation.samplers) - 1
                    
                    # Create animation channel for position
                    pos_channel = AnimationChannel(
                        sampler=position_sampler_idx,
                        target=AnimationChannelTarget(
                            node=joint_nodes[vrm_joint],
                            path="translation"
                        )
                    )
                    animation.channels.append(pos_channel)
                
                # For all joints, create rotation animation
                # Find child joints to determine rotation
                children = hierarchy.get(vrm_joint, [])
                if not children:
                    logger.info(f"Joint {vrm_joint} has no children, skipping rotation animation")
                    continue
                    
                # Create quaternion data for the joint
                rotation_data = np.zeros((n_frames, 4), dtype=np.float32)
                
                # Track min/max values for the accessor
                rot_min = [float('inf')] * 4
                rot_max = [float('-inf')] * 4
    
    # Process each frame
                for frame in range(n_frames):
                    joint_pos = joint_data[frame]
                    
                    # Different rotation calculations based on joint type
                    quaternion = None
                    
                    # For joints with one child, create a simple rotation that points to the child
                    if len(children) == 1:
                        child = children[0]
                        child_mdm_joint = VRM_BONE_MAPPING.get(child)
                        
                        if child_mdm_joint and child_mdm_joint in joint_positions:
                            child_pos = joint_positions[child_mdm_joint]['data'][frame]
                            
                            # Calculate direction vector from joint to child
                            direction = child_pos - joint_pos
                            
                            # Create quaternion looking at the child joint
                            # Convert to VRM coordinate system
                            direction_vrm = np.array([direction[0], direction[1], -direction[2]])
                            
                            # Only compute quaternion if the direction has meaningful length
                            if np.linalg.norm(direction_vrm) > 0.01:
                                quaternion = look_at_quaternion(direction_vrm)
                            else:
                                # Use identity quaternion for tiny movements
                                quaternion = np.array([0, 0, 0, 1])
                    
                    # For joints with multiple children, use the average direction
                    elif len(children) > 1:
                        directions = []
                        for child in children:
                            child_mdm_joint = VRM_BONE_MAPPING.get(child)
                            if child_mdm_joint and child_mdm_joint in joint_positions:
                                child_pos = joint_positions[child_mdm_joint]['data'][frame]
                                direction = child_pos - joint_pos
                                
                                # Only use meaningful directions
                                if np.linalg.norm(direction) > 0.01:
                                    directions.append(direction)
                        
                        if directions:
                            # Use average direction
                            avg_direction = np.mean(directions, axis=0)
                            
                            # Convert to VRM coordinate system
                            direction_vrm = np.array([avg_direction[0], avg_direction[1], -avg_direction[2]])
                            
                            if np.linalg.norm(direction_vrm) > 0.01:
                                quaternion = look_at_quaternion(direction_vrm)
                            else:
                                quaternion = np.array([0, 0, 0, 1])
                        else:
                            quaternion = np.array([0, 0, 0, 1])
                    
                    # If we couldn't compute a quaternion, use identity
                    if quaternion is None:
                        quaternion = np.array([0, 0, 0, 1])
                    
                    # Ensure quaternion is normalized
                    quaternion = normalize(quaternion)
                    
                    # Store quaternion [x, y, z, w]
                    rotation_data[frame] = quaternion
                    
                    # Update min/max
                    for i in range(4):
                        rot_min[i] = min(rot_min[i], float(quaternion[i]))
                        rot_max[i] = max(rot_max[i], float(quaternion[i]))
                
                # Convert rotation data to bytes
                rotation_bytes = rotation_data.tobytes()
                
                # Create buffer view for rotation data
                rot_buffer_view = BufferView(
                    buffer=0,
                    byteOffset=len(binary_data),
                    byteLength=len(rotation_bytes),
                    target=ARRAY_BUFFER
                )
                gltf.bufferViews.append(rot_buffer_view)
                rotation_buffer_view_idx = len(gltf.bufferViews) - 1
                
                # Add rotation data to binary blob
                binary_data.extend(rotation_bytes)
                
                # Create accessor for rotation data
                rotation_accessor = Accessor(
                    bufferView=rotation_buffer_view_idx,
                    componentType=FLOAT,
                    count=n_frames,
                    type=VEC4,
                    min=rot_min,
                    max=rot_max
                )
                gltf.accessors.append(rotation_accessor)
                rotation_accessor_idx = len(gltf.accessors) - 1
                
                # Create sampler for the rotation animation
                rot_sampler = AnimationSampler(
                    input=time_accessor_idx,
                    output=rotation_accessor_idx,
                    interpolation="LINEAR"
                )
                animation.samplers.append(rot_sampler)
                rotation_sampler_idx = len(animation.samplers) - 1
                
                # Create animation channel for rotation
                rot_channel = AnimationChannel(
                    sampler=rotation_sampler_idx,
                    target=AnimationChannelTarget(
                        node=joint_nodes[vrm_joint],
                        path="rotation"
                    )
                )
                animation.channels.append(rot_channel)
                
            except Exception as e:
                logger.error(f"Error processing joint {vrm_joint}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

        # Add animation to GLTF
        logger.info("Adding animation to GLTF")
        gltf.animations.append(animation)
        
        # Create buffer
        logger.info(f"Creating buffer with {len(binary_data)} bytes of data")
        buffer = Buffer(
            byteLength=len(binary_data)
        )
        gltf.buffers.append(buffer)
        
        # Debug GLTF structure
        logger.info("=== GLTF Structure Debug ===")
        logger.info(f"Scenes: {len(gltf.scenes)}")
        logger.info(f"Nodes: {len(gltf.nodes)}")
        if gltf.nodes and gltf.nodes[0].children:
            logger.info(f"Node children example: {gltf.nodes[0].children}")
        logger.info(f"Animations: {len(gltf.animations)}")
        if gltf.animations:
            logger.info(f"First animation channels: {len(gltf.animations[0].channels)}")
            logger.info(f"First animation samplers: {len(gltf.animations[0].samplers)}")
            if gltf.animations[0].channels:
                logger.info(f"First channel sampler: {gltf.animations[0].channels[0].sampler}")
                logger.info(f"First channel target: {vars(gltf.animations[0].channels[0].target)}")
        logger.info(f"Buffers: {len(gltf.buffers)}")
        if len(gltf.buffers) > 0:
            logger.info(f"First buffer length: {gltf.buffers[0].byteLength}")
        logger.info(f"Binary data length: {len(binary_data)}")
        logger.info(f"BufferViews: {len(gltf.bufferViews)}")
        logger.info(f"Accessors: {len(gltf.accessors)}")
        logger.info("=== End GLTF Structure Debug ===")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Set the binary data
        logger.info("Setting binary data for GLB file")
        gltf.set_binary_blob(bytes(binary_data))
        
        # Save the GLB file
        logger.info(f"Saving GLB file to: {output_path}")
        gltf.save_binary(output_path)
        logger.info("save_binary completed")
        
        # Check if file was created
        if os.path.exists(output_path):
            logger.info(f"Successfully created GLB file: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            raise ValueError(f"GLB file was not created at {output_path}")
            
        # Generate MP4 visualization
        mp4_path = output_path.replace('.glb', '.mp4')
        logger.info(f"Generating MP4 visualization: {mp4_path}")
        
        try:
            # Check required packages
            logger.info("Checking required packages for MP4 generation...")
            try:
                import matplotlib
                logger.info(f"matplotlib version: {matplotlib.__version__}")
                matplotlib.use('Agg')
            except ImportError as e:
                logger.error(f"matplotlib not found: {e}")
                raise
                
            try:
                from moviepy.editor import VideoClip
                import moviepy
                logger.info("moviepy package found and imported successfully")
            except ImportError as e:
                logger.error(f"moviepy not found: {e}")
                raise
                
            # Get motion data in correct format for visualization
            # Original motion_data is [batch_size, n_joints, 3, n_frames]
            # We need [n_frames, n_joints, 3]
            logger.info(f"Motion data shape before visualization: {motion_data.shape}")
            
            # Create MP4 animation
            logger.info("Setting up visualization parameters")
            skeleton = paramUtil.t2m_kinematic_chain
            logger.info(f"Skeleton chain: {skeleton}")
            title = "Motion Visualization"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
            logger.info(f"Output directory created/verified: {os.path.dirname(mp4_path)}")
            
            # Check if FFMPEG is available
            try:
                from moviepy.config import get_setting
                ffmpeg_path = get_setting("FFMPEG_BINARY")
                logger.info(f"FFMPEG binary path: {ffmpeg_path}")
                if not os.path.exists(ffmpeg_path):
                    logger.warning(f"FFMPEG not found at {ffmpeg_path}, MP4 generation may fail")
            except Exception as e:
                logger.warning(f"Could not verify FFMPEG: {e}")
            
            logger.info("Calling plot_3d_motion...")
            animation = plot_3d_motion(mp4_path, skeleton, motion_data, title=title, dataset='humanml', fps=fps)
            logger.info(f"plot_3d_motion returned: {animation}")
            
            if animation is not None:
                logger.info("Writing animation to MP4 file...")
                try:
                    # Set the duration based on the number of frames and fps
                    animation.duration = len(motion_data) / fps
                    
                    # Write the video file with progress bar
                    animation.write_videofile(
                        mp4_path,
                        fps=fps,
                        codec='libx264',
                        bitrate='5000k',
                        audio=False,
                        logger=None  # Disable moviepy's console output
                    )
                    animation.close()
                    
                    if os.path.exists(mp4_path):
                        logger.info(f"Successfully created MP4 file: {mp4_path} ({os.path.getsize(mp4_path)} bytes)")
                    else:
                        logger.error("MP4 file was not created after write_videofile call")
                except Exception as write_error:
                    logger.error(f"Error writing video file: {str(write_error)}")
                    import traceback
                    logger.error(f"Video write traceback: {traceback.format_exc()}")
            else:
                logger.warning("plot_3d_motion returned None, skipping MP4 generation")
                
            if not os.path.exists(mp4_path):
                logger.warning(f"MP4 file was not created at expected path: {mp4_path}")
                # Check for other potential files
                dir_path = os.path.dirname(mp4_path)
                base_name = os.path.splitext(os.path.basename(mp4_path))[0]
                for file in os.listdir(dir_path):
                    if file.startswith(base_name):
                        logger.info(f"Found related file: {file}")
                        
                # Check if FFMPEG is the issue
                if not os.path.exists(ffmpeg_path):
                    logger.error("MP4 generation likely failed due to missing FFMPEG. Please install FFMPEG and add it to your PATH.")
        except Exception as e:
            logger.error(f"Error during MP4 generation: {str(e)}")
            import traceback
            logger.error(f"MP4 generation traceback: {traceback.format_exc()}")
            logger.warning("Continuing despite MP4 generation failure")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in motion_to_glb: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to convert motion to GLB: {str(e)}") 
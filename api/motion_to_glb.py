import numpy as np
import os
from pathlib import Path
from pygltflib import GLTF2, Node, Scene, Buffer, BufferView, Accessor, Animation, AnimationChannel, AnimationSampler
from pygltflib import Mesh, Primitive, Asset, Scene, Node, Buffer, BufferView, Accessor, AnimationChannelTarget
from pygltflib import ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER, SCALAR, VEC3, VEC4, FLOAT, UNSIGNED_SHORT
import struct
import logging

logger = logging.getLogger(__name__)

# VRM bone mapping to MDM joints
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

def create_buffer_from_data(data, gltf):
    """Helper function to create a buffer from data"""
    try:
        logger.info(f"Creating buffer from data of length {len(data)}")
        buffer = struct.pack('<{}f'.format(len(data)), *data)
        logger.info(f"Created packed buffer of size {len(buffer)} bytes")
        
        # Get current offset in the buffer
        current_offset = 0 if not hasattr(gltf, '_binary_data') else len(gltf._binary_data)
        
        # Extend the binary data
        if not hasattr(gltf, '_binary_data'):
            gltf._binary_data = buffer
        else:
            gltf._binary_data += buffer
            
        # Update buffer length
        if len(gltf.buffers) == 0:
            gltf.buffers.append(Buffer())
        gltf.buffers[0].byteLength = len(gltf._binary_data)
        
        # Create buffer view
        buffer_view = BufferView(
            buffer=0,
            byteOffset=current_offset,
            byteLength=len(buffer),
            target=ARRAY_BUFFER
        )
        gltf.bufferViews.append(buffer_view)
        
        return len(gltf.bufferViews) - 1
    except Exception as e:
        logger.error(f"Error in create_buffer_from_data: {str(e)}")
        raise

def create_accessor(buffer_view_idx, count, component_type, type, gltf):
    """Helper function to create an accessor"""
    accessor = Accessor(
        bufferView=buffer_view_idx,
        componentType=component_type,
        count=count,
        type=type
    )
    gltf.accessors.append(accessor)
    return len(gltf.accessors) - 1

def motion_to_glb(motion_data, output_path, fps=30):
    """
    Convert motion data to GLB format with VRM-compatible skeleton
    
    Args:
        motion_data (np.ndarray): Motion data array from MDM with shape (batch_size, n_joints, 3, n_frames)
        output_path (str): Path to save the GLB file
        fps (int): Frames per second for the animation
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
        
        if n_dims != 3:
            raise ValueError(f"Expected 3 dimensions per joint, got {n_dims}")
            
        # Take first batch and transpose to (n_frames, n_joints, 3)
        logger.info("Reshaping motion data")
        motion_data = motion_data[0].transpose(2, 0, 1)
        logger.info(f"Reshaped motion data shape: {motion_data.shape}")

        # Create GLTF structure
        logger.info("Creating GLTF structure")
        gltf = GLTF2()
        gltf.scene = 0
        gltf.scenes = [Scene(nodes=[0])]  # Root node
        gltf.asset = Asset(version="2.0", generator="MDM-to-GLB Converter")

        # Initialize empty lists
        gltf.nodes = []
        gltf.meshes = []
        gltf.animations = []
        gltf.bufferViews = []
        gltf.accessors = []
        gltf.buffers = []  # Will be populated in create_buffer_from_data
        gltf._binary_data = b''  # Initialize empty binary data

        # Create skeleton nodes with proper transforms
        logger.info("Creating skeleton nodes")
        joint_nodes = {}
        
        # Create Armature node first
        armature_node = Node(
            name="Armature",
            translation=[0, 0, 0],
            rotation=[0, 0, 0, 1],
            scale=[1, 1, 1],
            children=[]
        )
        gltf.nodes.append(armature_node)
        armature_idx = 0

        # Create all bone nodes with identity transforms
        for joint_name in VRM_BONE_MAPPING.keys():
            node = Node(
                name=joint_name,
                translation=[0, 0, 0],
                rotation=[0, 0, 0, 1],  # Identity quaternion
                scale=[1, 1, 1],
                children=[]
            )
            gltf.nodes.append(node)
            joint_nodes[joint_name] = len(gltf.nodes) - 1

        # Set up parent-child relationships
        logger.info("Setting up skeleton hierarchy")
        
        # First, make all bones children of the Armature
        armature_node.children = list(joint_nodes.values())
        
        # Then set up bone hierarchy
        for joint_name in VRM_BONE_MAPPING.keys():
            current_idx = joint_nodes[joint_name]
            
            if joint_name == 'hips':
                continue  # hips is directly under Armature
                
            parent = None
            if 'Shoulder' in joint_name:
                parent = 'chest'
            elif 'UpperArm' in joint_name:
                parent = joint_name.replace('Upper', 'Shoulder')
            elif 'LowerArm' in joint_name:
                parent = joint_name.replace('Lower', 'Upper')
            elif 'Hand' in joint_name:
                parent = joint_name.replace('Hand', 'LowerArm')
            elif 'UpperLeg' in joint_name:
                parent = 'hips'
            elif 'LowerLeg' in joint_name:
                parent = joint_name.replace('Lower', 'Upper')
            elif 'Foot' in joint_name:
                parent = joint_name.replace('Foot', 'LowerLeg')
            elif joint_name == 'spine':
                parent = 'hips'
            elif joint_name == 'chest':
                parent = 'spine'
            elif joint_name == 'neck':
                parent = 'chest'
            elif joint_name == 'head':
                parent = 'neck'
                
            if parent and parent in joint_nodes:
                parent_idx = joint_nodes[parent]
                if current_idx not in gltf.nodes[parent_idx].children:
                    gltf.nodes[parent_idx].children.append(current_idx)

        # Create animation
        logger.info(f"Creating animation timeline with {n_frames} frames at {fps} FPS")
        times = np.linspace(0, n_frames / fps, n_frames, dtype=np.float32)
        
        animation = Animation(
            name="motion",
            channels=[],
            samplers=[]
        )

        # Process each joint's animation
        logger.info("Processing joint animations")
        for joint_name, mdm_joint in VRM_BONE_MAPPING.items():
            try:
                joint_idx = list(VRM_BONE_MAPPING.values()).index(mdm_joint)
                if joint_idx >= n_joints:
                    logger.warning(f"Skipping joint {joint_name} (index {joint_idx} >= {n_joints})")
                    continue

                logger.info(f"Processing joint: {joint_name} (MDM joint: {mdm_joint})")
                rotations = motion_data[:, joint_idx]
                
                # Create time buffer
                time_accessor_idx = create_accessor(
                    create_buffer_from_data(times, gltf),
                    n_frames,
                    FLOAT,
                    SCALAR,
                    gltf
                )

                # Create rotation buffer with normalized quaternions
                rotation_data = []
                for frame in range(n_frames):
                    x, y, z = rotations[frame]
                    # Convert Euler angles to quaternion
                    cx = np.cos(x * 0.5)
                    sx = np.sin(x * 0.5)
                    cy = np.cos(y * 0.5)
                    sy = np.sin(y * 0.5)
                    cz = np.cos(z * 0.5)
                    sz = np.sin(z * 0.5)
                    
                    qw = cx * cy * cz + sx * sy * sz
                    qx = sx * cy * cz - cx * sy * sz
                    qy = cx * sy * cz + sx * cy * sz
                    qz = cx * cy * sz - sx * sy * cz
                    
                    # Normalize quaternion
                    length = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                    qw /= length
                    qx /= length
                    qy /= length
                    qz /= length
                    
                    rotation_data.extend([qx, qy, qz, qw])

                rotation_accessor_idx = create_accessor(
                    create_buffer_from_data(rotation_data, gltf),
                    n_frames,
                    FLOAT,
                    VEC4,
                    gltf
                )

                # Create animation sampler
                sampler = AnimationSampler(
                    input=time_accessor_idx,
                    output=rotation_accessor_idx,
                    interpolation="LINEAR"
                )
                animation.samplers.append(sampler)
                sampler_idx = len(animation.samplers) - 1

                # Create animation channel
                target = AnimationChannelTarget(
                    node=joint_nodes[joint_name],
                    path="rotation"
                )
                channel = AnimationChannel(
                    sampler=sampler_idx,
                    target=target
                )
                animation.channels.append(channel)
                
            except Exception as e:
                logger.error(f"Error processing joint {joint_name}: {str(e)}")
                raise

        logger.info("Adding animation to GLTF")
        gltf.animations.append(animation)

        # Debug log GLTF structure
        logger.info("=== GLTF Structure Debug ===")
        logger.info(f"Scenes: {len(gltf.scenes)}")
        logger.info(f"Nodes: {len(gltf.nodes)}")
        logger.info(f"Node children example: {gltf.nodes[0].children if gltf.nodes else 'No nodes'}")
        logger.info(f"Animations: {len(gltf.animations)}")
        if gltf.animations:
            logger.info(f"First animation channels: {len(gltf.animations[0].channels)}")
            logger.info(f"First animation samplers: {len(gltf.animations[0].samplers)}")
            if gltf.animations[0].channels:
                first_channel = gltf.animations[0].channels[0]
                logger.info(f"First channel sampler: {first_channel.sampler}")
                logger.info(f"First channel target: {{'node': {first_channel.target.node}, 'path': '{first_channel.target.path}'}}")
        logger.info(f"Buffers: {len(gltf.buffers)}")
        if gltf.buffers:
            logger.info(f"First buffer length: {gltf.buffers[0].byteLength}")
            logger.info(f"Binary data length: {len(gltf._binary_data)}")
        logger.info(f"BufferViews: {len(gltf.bufferViews)}")
        logger.info(f"Accessors: {len(gltf.accessors)}")
        logger.info("=== End GLTF Structure Debug ===")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as GLB
        logger.info(f"Saving GLB file to: {output_path}")
        try:
            # Add debug info about buffer before saving
            logger.info(f"Binary data before save - type: {type(gltf._binary_data)}, length: {len(gltf._binary_data)}")
            logger.info(f"Buffer byteLength: {gltf.buffers[0].byteLength}")
            logger.info(f"First few bytes: {gltf._binary_data[:20] if len(gltf._binary_data) > 20 else gltf._binary_data}")
            
            # Set the binary blob for saving
            gltf.set_binary_blob(gltf._binary_data)
            
            # Save the GLB file
            gltf.save_binary(output_path)
            logger.info("save_binary completed")
            
        except Exception as e:
            logger.error(f"Error during save_binary: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        if os.path.exists(output_path):
            logger.info(f"Successfully created GLB file: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            raise ValueError(f"GLB file was not created at {output_path}")
        
        return output_path

    except Exception as e:
        logger.error(f"Error in motion_to_glb: {str(e)}")
        raise ValueError(f"Failed to convert motion to GLB: {str(e)}") 
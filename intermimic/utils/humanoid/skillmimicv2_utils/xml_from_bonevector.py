import pickle
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom


# Load bone vectors
scene_number = 110
with open(f'intermimic/data/assets/parahome/seq/s{scene_number}/bone_vectors.pkl', 'rb') as f:
    bone_vectors = pickle.load(f)

# Joint information
body_vector = bone_vectors.get('body', {})
lhand_vector = bone_vectors.get('lhand', {})
rhand_vector = bone_vectors.get('rhand', {})

# Create MuJoCo XML structure
mujoco = Element('mujoco', model='humanoid')
compiler = SubElement(mujoco, 'compiler', coordinate='local')
statistic = SubElement(mujoco, 'statistic', extent='2', center='0 0 1')
option = SubElement(mujoco, 'option', timestep='0.00555')

# Worldbody
worldbody = SubElement(mujoco, 'worldbody')
light = SubElement(worldbody, 'light', cutoff='100', diffuse='1 1 1', dir='-0 0 -1.3', directional='true', exponent='1', pos='0 0 1.3', specular='.1 .1 .1')
floor = SubElement(worldbody, 'geom', conaffinity='1', condim='3', name='floor', pos='0 0 0', rgba='0.8 0.9 0.8 1', size='100 100 .2', type='plane', material='MatPlane')

# Helper function to create a body with joints and geometry
def create_body(parent, name, length, pos, geom_type='capsule'):
    body = SubElement(parent, 'body', name=name, pos=' '.join(map(str, pos)))
    SubElement(body, 'geom', type=geom_type, size=str(length), pos='0 0 0')
    SubElement(body, 'joint', name=f"{name}_x", type='hinge', pos='0 0 0', axis='1 0 0', stiffness='500', damping='50', armature='0.02', range='-180.0000 180.0000')
    SubElement(body, 'joint', name=f"{name}_y", type='hinge', pos='0 0 0', axis='0 1 0', stiffness='500', damping='50', armature='0.02', range='-180.0000 180.0000')
    SubElement(body, 'joint', name=f"{name}_z", type='hinge', pos='0 0 0', axis='0 0 1', stiffness='500', damping='50', armature='0.02', range='-180.0000 180.0000')
    return body

# Define which joints should be spheres
sphere_joints = {'jL5S1', 'jL4L3', 'jL1T12', 'jT9T8', 'jC1Head'}

# Build body hierarchy
bodies = {}
all_bone_data = {**body_vector, **lhand_vector, **rhand_vector}

for parent_name, children_dict in all_bone_data.items():
    # Create parent body if it doesn't exist
    if parent_name not in bodies:
        if parent_name == 'pHipOrigin':
            bodies[parent_name] = SubElement(worldbody, 'body', name='Pelvis', pos='0 0 0')
            SubElement(bodies[parent_name], 'freejoint', name='Pelvis')
        else:
            # For parent bodies, use a default size since we don't have length data for them
            geom_type = 'sphere' if parent_name in sphere_joints else 'capsule'
            bodies[parent_name] = create_body(worldbody, parent_name, 0.05, (0, 0, 0), geom_type)

    # Create child bodies
    for child_name, bone_vector in children_dict.items():
        if child_name not in bodies:
            # Calculate bone length from the vector
            bone_length = (bone_vector[0]**2 + bone_vector[1]**2 + bone_vector[2]**2)**0.5
            # Use the bone vector as the relative position
            pos = (bone_vector[0], bone_vector[1], bone_vector[2])

            geom_type = 'sphere' if child_name in sphere_joints else 'capsule'
            bodies[child_name] = create_body(bodies[parent_name], child_name, bone_length/2, pos, geom_type)

# Format XML output
xml_str = xml.dom.minidom.parseString(tostring(mujoco)).toprettyxml(indent="  ")

# Create output directory if it doesn't exist
import os
output_dir = "intermimic/data/assets/parahome/sim_raw_human"
os.makedirs(output_dir, exist_ok=True)

output_file = f"{output_dir}/s{scene_number}.xml"
with open(output_file, "w") as f:
    f.write(xml_str)

print(f"XML file '{output_file}' created successfully.")

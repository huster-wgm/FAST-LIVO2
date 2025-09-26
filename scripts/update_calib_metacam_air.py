import os
import yaml
import json
import numpy as np


class Intrinsic:
    def __init__(self, model: str, fx: float, fy: float, cx: float, cy: float, dist: list):
        self.model = model
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dist = dist

    def to_matrix(self):
        return [
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ]

    def resize(self, scale: float):
        self.fx *= scale
        self.fy *= scale
        self.cx *= scale
        self.cy *= scale


class Extrinsic:
    def __init__(self, rotation: np.array, translation: np.array):
        self.rotation = rotation # 3x3 rotation matrix
        self.translation = translation # 3x1 translation vector
        
    def to_matrix(self):
        """generate 4x4 transformation matrix from rotation and translation
        """
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = self.rotation
        extrinsic_matrix[:3, 3] = self.translation.flatten()
        return extrinsic_matrix


def parse_calib_json(json_file_path):
    with open(json_file_path, "r") as f:
        calib_json = json.load(f)

    cameras = {}
    for cam in calib_json["cameras"]:
        # Parse distortion parameters
        dist = []
        if "distortion" in cam and "params" in cam["distortion"]:
            params = cam["distortion"]["params"]
            dist = [params.get("k1", 0), params.get("k2", 0), 
                   params.get("k3", 0), params.get("k4", 0)]
        
        intrinsic = Intrinsic(
            model=cam["type"],
            fx=cam["intrinsic"]["fl_x"],
            fy=cam["intrinsic"]["fl_y"],
            cx=cam["intrinsic"]["cx"],
            cy=cam["intrinsic"]["cy"],
            dist=dist
        )
        rotation = np.array(cam["transform_from_lidar"]["rotation"]).reshape(3, 3)
        translation = np.array(cam["transform_from_lidar"]["position"]).reshape(3, 1)
        extrinsic = Extrinsic(rotation, translation)
        cameras[cam["name"]] = {
            "width": cam["width"],
            "height": cam["height"],
            "intrinsic": intrinsic,
            "extrinsic": extrinsic
        }

    return cameras



def update_camera_yaml(camera: dict, yaml_file_path: str):
    # Update the YAML data with new calibration info
    intrinsic = camera["intrinsic"]
    scale = 0.25  # Scale factor for resizing
    
    # Create ordered dictionary to maintain the exact order
    yaml_data = {
        "cam_model": "Pinhole",
        "scale": 1.0,
        "cam_width": int(camera["width"] * scale),
        "cam_height": int(camera["height"] * scale),
        "cam_fx": intrinsic.fx * scale,
        "cam_fy": intrinsic.fy * scale,
        "cam_cx": intrinsic.cx * scale,
        "cam_cy": intrinsic.cy * scale,
        "cam_d0": 0,
        "cam_d1": 0,
        "cam_d2": 0,
        "cam_d3": 0
    }

    # Write the updated calibration data to the YAML file with preserved order
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"Updated calibration data written to {yaml_file_path}")
    

def update_fastlivo2_config_yaml(camera: dict, yaml_file_path: str):
    # Load existing YAML data
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Ensure extrin_calib exists
    if 'extrin_calib' not in yaml_data:
        yaml_data['extrin_calib'] = {}

    # Custom write function to maintain formatting
    def write_yaml(data, file):
        def format_list(lst):
            return "[" + ", ".join(f"{x:.7g}" for x in lst) + "]"

        # Write each section
        for key, value in data.items():
            if isinstance(value, dict):
                file.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        file.write(f"  {subkey}: {format_list(subvalue)}\n")
                    else:
                        file.write(f"  {subkey}: {subvalue}\n")
            else:
                file.write(f"{key}: {value}\n")

    # Update values while maintaining the list format
    yaml_data['extrin_calib']['Rcl'] = camera['extrinsic'].rotation.flatten().tolist()
    yaml_data['extrin_calib']['Pcl'] = camera['extrinsic'].translation.flatten().tolist()

    # Update img_topic
    tname = 'left' if use_left_cam else 'right'
    yaml_data['common']['img_topic'] = f"/camera/{tname}/jpeg_1k/undistort"

    # Write the updated YAML data back to the file
    with open(yaml_file_path, "w") as f:
        write_yaml(yaml_data, f)



def update_preprocess_config_yaml(cameras: dict, yaml_file_path: str):
    # Load existing YAML data
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Update camera parameters for left and right cameras
    camera_mapping = {'left': 'cam0', 'right': 'cam1'}
    
    # Update refer_cam_id based on global use_left_cam
    yaml_data['refer_cam_id'] = 0 if use_left_cam else 1

    for cam_name, yaml_cam_key in camera_mapping.items():
        if cam_name in cameras:
            cam = cameras[cam_name]
            intrinsic = cam['intrinsic']
            extrinsic = cam['extrinsic']
            
            # Update T_lidar_cam as a flattened list
            yaml_data[yaml_cam_key]['T_lidar_cam'] = extrinsic.to_matrix().flatten().tolist()
            
            # Update intrinsics [fu, fv, cu, cv]
            yaml_data[yaml_cam_key]['intrinsics'] = [
                intrinsic.fx, intrinsic.fy,
                intrinsic.cx, intrinsic.cy
            ]
            
            # Update distortion coefficients
            yaml_data[yaml_cam_key]['distortion_coeffs'] = intrinsic.dist
            
            # Update resolution
            yaml_data[yaml_cam_key]['in_resolution'] = [
                int(cam['width']), int(cam['height'])
            ]

            yaml_data[yaml_cam_key]['out_resolution'] = [
                int(cam['width'] * 0.25), int(cam['height'] * 0.25)
            ]

    # Custom write function to maintain formatting
    def write_yaml(data, file):
        def format_matrix(mat):
            if len(mat) == 16:  # 4x4 matrix
                rows = [mat[i:i+4] for i in range(0, 16, 4)]
                formatted = []
                for row in rows:
                    formatted.append(", ".join(f"{x:.16g}" for x in row))
                return "[\n    " + ",\n    ".join(formatted) + "]"
            else:  # Regular list
                return "[" + ", ".join(f"{x:.16g}" for x in mat) + "]"

        def write_section(section, indent=""):
            for key, value in section.items():
                if isinstance(value, dict):
                    file.write(f"{indent}{key}:\n")
                    write_section(value, indent + "  ")
                elif isinstance(value, list):
                    if key in ['T_lidar_cam', 'T_imu_lidar']:
                        file.write(f"{indent}{key}: {format_matrix(value)}\n")
                    else:
                        file.write(f"{indent}{key}: {format_matrix(value)}\n")
                else:
                    file.write(f"{indent}{key}: {value}\n")

        write_section(data)

    # Write the updated YAML data back to the file
    with open(yaml_file_path, "w") as f:
        write_yaml(yaml_data, f)



if __name__ == "__main__":
    global use_left_cam
    use_left_cam = True
    fastlivo_ws = "/home/intel/workspaces/catkin_ws/src"
    # Load the calibration data from the YAML file
    cameras = parse_calib_json("metacam/2025-08-07_16-05-41/info/calibration.json")
    
    # update setting in edu_camera.yaml and edu.yaml
    camera = cameras['left'] if use_left_cam else cameras['right']
    update_camera_yaml(camera, f"{fastlivo_ws}/FAST-LIVO2/config/edu_camera.yaml")
    update_fastlivo2_config_yaml(camera, f"{fastlivo_ws}/FAST-LIVO2/config/edu.yaml")
    
    # update setting in data_preprocessing
    update_preprocess_config_yaml(cameras, f"{fastlivo_ws}/data_preprocessing/config/data_preprocessing.yaml")
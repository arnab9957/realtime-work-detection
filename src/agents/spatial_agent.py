import math
from typing import Dict, Optional, Tuple
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    SpatialMetrics,
    Vector3D
)

class SpatialAgent:
    """Agent responsible for accurate spatial calculations, like point-to-box distances."""
    
    def __init__(self):
        # We can store state if we need smoothing or velocity, but for now it's purely functional.
        pass

    def evaluate_spatial_metrics(
        self,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject]
    ) -> SpatialMetrics:
        """
        Calculates spatial relationships between the astronaut's hands and the objects.
        """
        metrics = SpatialMetrics()
        
        # Determine which wrist to use
        wrist_key = None
        if "wrist" in pose.keypoints_2d:
            wrist_key = "wrist"
        elif "right_wrist" in pose.keypoints_2d:
            wrist_key = "right_wrist"
        
        wrist_pos_3d = None
        if pose.joints and "wrist" in pose.joints:
            wrist_pos_3d = pose.joints["wrist"].pos_rack
        
        cont = objects.get("container_box")
        if cont:
            # 1. Container distance
            if cont.bbox and wrist_key:
                wx, wy, _ = pose.keypoints_2d[wrist_key]
                
                # Check if wrist is inside the 2D bounding box
                if (cont.bbox.xmin <= wx <= cont.bbox.xmax and 
                    cont.bbox.ymin <= wy <= cont.bbox.ymax):
                    metrics.wrist_in_container_2d = True
                    metrics.distance_to_container_m = 0.0
                else:
                    metrics.wrist_in_container_2d = False
                    
                    # Shortest distance from wrist to container bbox edges in pixels
                    dx_pix = max(0.0, cont.bbox.xmin - wx, wx - cont.bbox.xmax)
                    dy_pix = max(0.0, cont.bbox.ymin - wy, wy - cont.bbox.ymax)
                    dist_pix = math.sqrt(dx_pix**2 + dy_pix**2)
                    
                    # Convert to meters
                    # Estimate pixels-per-meter based on 3D distance vs 2D distance to center
                    cont_cx, cont_cy = cont.bbox.centroid
                    center_dist_pix = math.sqrt((wx - cont_cx)**2 + (wy - cont_cy)**2)
                    
                    if wrist_pos_3d and center_dist_pix > 0:
                        center_dist_m = wrist_pos_3d.distance_to(cont.pos_rack)
                        m_per_pix = center_dist_m / center_dist_pix
                    else:
                        m_per_pix = 0.001 # rough fallback (1mm per px)
                        
                    metrics.distance_to_container_m = dist_pix * m_per_pix
            else:
                # Fallback if no 2D bbox or wrist point
                if wrist_pos_3d:
                    metrics.distance_to_container_m = wrist_pos_3d.distance_to(cont.pos_rack)
                else:
                    metrics.distance_to_container_m = 999.0
        
        # 2. Components distance
        for obj_name, obj in objects.items():
            if obj_name in ("container_box", "operator_hand", "hand", "person", "astronaut", "human_body"):
                continue
                
            if wrist_pos_3d:
                # For small items, center-to-center is generally sufficient
                metrics.distance_to_components_m[obj_name] = wrist_pos_3d.distance_to(obj.pos_rack)
            else:
                metrics.distance_to_components_m[obj_name] = 999.0
                
        return metrics

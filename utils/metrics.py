import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def calcAngularDistance(gt_rot, pr_rot , degrees = False):
    """Calculates the angular distance between two rotation matrices;
    """ 
    rotDiff = np.matmul(gt_rot, np.transpose(pr_rot, (0,2,1)))
    trace = np.trace(rotDiff,axis1 = 1 , axis2 = 2 )
    if degrees:
        return np.rad2deg(np.arccos((trace-1.0)/2.0))
    else:
        return np.arccos((trace-1.0)/2.0)


def calcNormalisedPositionDistance(gt_pos,pr_pos):
    
    l2_norm = np.linalg.norm(gt_pos - pr_pos , axis = 1)
    
    return l2_norm / np.linalg.norm(gt_pos , axis = 1)


def evaluate_pose_add(gt_rot,pr_rot,gt_pos,pr_pos):
    """
    An estimated pose is correct if the average distance 
    is smaller than 5pixels.k◦,  k  cm as proposed  in  Shotton  et  al.  (2013).
    The  5◦,5cm metric considers an estimated pose to be correct if
    its rotation error is within 5◦ and the translation error is below 5cm.
    Provide also the results with 2◦, 2cm and 10◦, 10 cm.
    input: calculates the error per batch size
    """
    b , _ , _  = gt_rot.shape
    #count_correct = {k: np.zeros((self.num_classes, num_iter), dtype=np.float32) for k in ["0.02", "0.05", "0.10"]}
    rotation_error = calcAngularDistance(gt_rot,pr_rot,degrees=True)
    translation_error = np.linalg.norm(gt_pos - pr_pos , axis = 1) * 100 #in cm
    #count correct
    #005
    condition_005 = np.where((rotation_error < 5) & (translation_error < 5), 1 ,0)
    
    count_005 = np.count_nonzero(condition_005)
    #002
    condition_002 = np.where((rotation_error < 2) & (translation_error < 2), 1 ,0)
    
    count_002 = np.count_nonzero(condition_002)
    #010
    condition_010 = np.where((rotation_error < 10) & (translation_error < 10), 1 ,0)
    
    count_010 = np.count_nonzero(condition_010)
    
    #calculate acuracy
    acc_005 = 100 * float(count_005) / b
    acc_002 = 100 * float(count_002) / b
    acc_010 = 100 * float(count_010) / b

    return acc_002 , acc_005 , acc_010

def add(pts,P_gt,P_pred):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    :param Pest: Estimated transformation matrix (i.e. from object to camera view) (bx4x4).
    :param Pgt: GT transformation matrix (bx4x4).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    Then  a  pose estimation is considered to be correct if the computed average distance is within 10% of the model diameter.
    """
    b , _ , _ = P_gt.size()
    #get homogeneous coordinates for vertices
    pts = torch.torch.nn.functional.pad(pts[:,:,],[0,1],"constant",1.0)
    pts = pts.transpose(2,1)

    pts_est = torch.matmul(P_pred,pts) # 

    pts_gt = torch.matmul(P_gt,pts)

    pts_est = pts_est.transpose(2,1)
    pts_gt = pts_gt.transpose(2,1)

    #convert points from homogeneous
    pts_est =  pts_est[..., :-1]
    pts_gt =  pts_gt[..., :-1]
    
    e = np.linalg.norm(pts_est.cpu() - pts_gt.cpu(), axis = 2).mean(axis = 1)

    #NOTE:Consider using another value
    diagonial = 0.354 #in meters

    threshold_002 = np.where( e < 0.02 * diagonial, 1 ,0)
    threshold_005 = np.where( e < 0.05 * diagonial, 1 ,0)
    threshold_010 = np.where( e < 0.1 * diagonial, 1 ,0)

    #count accepted poses
    accepted_002 = np.count_nonzero(threshold_002)
    accepted_005 = np.count_nonzero(threshold_005)
    accepted_010 = np.count_nonzero(threshold_010)

    #calculate acuracy
    acc_002 = 100 * float(accepted_002) / b
    acc_005 = 100 * float(accepted_005) / b
    acc_010 = 100 * float(accepted_010) / b
    
    return acc_002 , acc_005 , acc_010


def ProjectionError(pxls_pred , pxls_gt):
    """
    2D  Projection: 
    focuses  on  the  matching  of  pose  esti-mation on 2D images. 
    This metric is considered to beimportant  for  applications  such  as  
    augmented  reality.
    We accept  a  pose estimation when the 2D projection error is smaller than 
    a predefined threshold.
    Input: 
    predicted pxls based on the estimated pose and gt pixels derived from gt pose
    """
    #NOTE: check thoroughly the logic
    b , _ , _ , w = pxls_pred.size()

    pxls_pred = pxls_pred.transpose(3,1)
    pxls_pred = pxls_pred.transpose(3,2)

    projection_error = np.linalg.norm(pxls_gt.cpu() - pxls_pred.cpu() , axis = 1).mean(axis = (1,2))

    threshold_002 = np.where(projection_error < 5, 1 ,0)
    
    threshold_005 = np.where(projection_error < 5, 1 ,0)

    threshold_010 = np.where(projection_error < 10, 1 ,0)

    #count accepted poses
    accepted_002 = np.count_nonzero(threshold_002)
    accepted_005 = np.count_nonzero(threshold_005)
    accepted_010 = np.count_nonzero(threshold_010)


    #calculate acuracy
    proj_002 = 100 * float(accepted_002) / b
    proj_005 = 100 * float(accepted_005) / b
    proj_010 = 100 * float(accepted_010) / b

    return proj_002 , proj_005 , proj_010



if __name__ == "__main__":
    #test metrics
    #Consider a counter-clockwise rotation of 90 degrees about the z-axis
    #r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    r1 = R.from_euler('z', 90., degrees=True)
    r2 = R.from_euler('z', 80, degrees=True)
    r3 = R.from_euler('z', 80, degrees=True)
    r1_ = R.from_euler('z', 81., degrees=True)
    r2_ = R.from_euler('z', 71, degrees=True)
    r3_ = R.from_euler('z', 80, degrees=True)
    #convert r to rotation matrix
    rotation_matrix_gt_1 = r1.as_matrix()
    rotation_matrix_pd_1 = r1_.as_matrix()
    rotation_matrix_gt_2 = r2.as_matrix()
    rotation_matrix_pd_2 = r2_.as_matrix()
    rotation_matrix_gt_3 = r3.as_matrix()
    rotation_matrix_pd_3 = r3_.as_matrix()

    rotation_matrix_gt = np.zeros((3,3,3))
    rotation_matrix_gt[0,:,:] = rotation_matrix_gt_1
    rotation_matrix_gt[1,:,:] = rotation_matrix_gt_2
    rotation_matrix_gt[2,:,:] = rotation_matrix_gt_3

    #
    rotation_matrix_pd = np.zeros((3,3,3))
    rotation_matrix_pd[0,:,:] = rotation_matrix_pd_1
    rotation_matrix_pd[1,:,:] = rotation_matrix_pd_2
    rotation_matrix_pd[2,:,:] = rotation_matrix_pd_3

    angular_loss = calcAngularDistance(rotation_matrix_gt,rotation_matrix_pd , False)
    #print()

    pos_1 = np.array([1,1,0.5],)
    pos_2 = np.array([0.9,1,0.5],)  
    pos_3 = np.array([12,0,0.0],)
    pos_4 = np.array([0.3,0,0.0],)
    pos_5 = np.array([0.25,0,0.0],)
    pos_6 = np.array([0.3,0,0.0],)

    gt_pos = np.zeros((3,3))
    gt_pos[0,:] = pos_1
    gt_pos[1,:] = pos_3
    gt_pos[2,:] = pos_5

    rt_pos = np.zeros((3,3))
    rt_pos[0,:] = pos_2
    rt_pos[1,:] = pos_4
    rt_pos[2,:] = pos_6

    pos_loss = calcNormalisedPositionDistance(gt_pos[:,:],rt_pos[:,:])

    total_loss = (pos_loss + angular_loss).mean()

    print(total_loss)

    print (evaluate_pose_add(rotation_matrix_gt,rotation_matrix_pd,gt_pos,rt_pos))
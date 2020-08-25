import visdom
import numpy
import torch
from utils import geometry
from math import radians
import cv2

try:
    from kaolin.graphics import DIBRenderer as Renderer
except ImportError:
    __KAOLIN_LOADED__ = False
    print("Error loading Kaolin.")
else:
    __KAOLIN_LOADED__ = True
    print("Kaoling loaded correctly.")

class NullVisualizer(object):
    def __init__(self):
        self.name = __name__

    def append_loss(self, epoch, global_iteration, loss, mode='train'):
        pass

    def show_images(self, images, title):
        pass

class VisdomVisualizer(object):
    def __init__(self, name, server="http://localhost", count=2):
        self.visualizer = visdom.Visdom(server=server, port=8097, env=name,\
            use_incoming_socket=False)
        self.name = name
        self.first_train_value = True
        self.first_test_value = True
        self.count = count
        self.plots = {}
        
    def append_loss(self, epoch, global_iteration, loss, loss_name="total", mode='train'):
        plot_name = loss_name + '_train_loss' if mode == 'train' else 'test_loss'
        opts = (
            {
                'title': plot_name,
                'xlabel': 'iterations',
                'ylabel': loss_name
            })
        loss_value = float(loss.detach().cpu().numpy())
        if loss_name not in self.plots:
            self.plots[loss_name] = self.visualizer.line(X=numpy.array([global_iteration]), Y=numpy.array([loss_value]), opts=opts)
        else:
            self.visualizer.line(X=numpy.array([global_iteration]), Y=numpy.array([loss_value]), win=self.plots[loss_name], name=mode, update = 'append')
        
    def show_images(self, images, title):
        b, c, h, w = images.size()
        recon_images = images.detach().cpu()[:self.count, [2, 1, 0], :, :]\
            if c == 3 else\
            images.detach().cpu()[:self.count, :, :, :]
        opts = (
        {
            'title': title, 'width': self.count / 2 * 320,
            'height': self.count / 4 * 240
        })
        self.visualizer.images(recon_images, opts=opts,\
            win=self.name + title + "_window")

    def show_map(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:self.count, :, :, :]
        for i in range(self.count):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :, :].squeeze(0)
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))

    def show_mask(self,images,title):
            #b, h, w , c = images.size()
            b, c, h , w = images.size()
            images = images.detach().cpu().numpy()[0]
            #print("img",images.size)
            # recon_images = images.detach().cpu()[:self.count, [2, 1, 0], :, :]\
            # if c == 3 else\
            # images.detach().cpu()[:self.count, :, :, :]
            opts = (
            {
            'title': title, 'width': self.count / 2 * 320,
            'height': self.count / 4 * 240
            })
            self.visualizer.images(images, opts=opts,\
                win=self.name + title + "_window")
    
    
    def show_both_keypoints(self,images,coordinates,predicted_coordinates,title):
        b, c, h , w = images.size()
        b , keypoints , _ = coordinates.shape
        final_images = numpy.zeros((b,h,w,c))
        radius = 5
        for i, image in enumerate(images):
            #process each image seperately
            im = (image.permute(1,2,0).cpu().numpy().copy() * 255).astype(numpy.int8)
            keypoint = coordinates[i] * h
            pr_keypoint = predicted_coordinates[i] * h
            for k, key in enumerate(keypoint):
                #if (k + 1) % 2 == 0:
                    #cv2.line(im, (int(point_1_x), int(point_1_y)), (int(keypoint[k][0]), int(keypoint[k][1])), (0,255,0), 2)
                #else:
                    #point_1_x = keypoint[k][0]
                    #point_1_y = keypoint[k][1]
                x = int(pr_keypoint[k][0])
                y = int(pr_keypoint[k][1])
                w = radius
                h = radius
                if k == 0:
                    #red --centroid
                    color = (255,0,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                    cv2.rectangle(im, (x, y), (x + w, y + h),color, -1)
                elif k == 1:
                    #blue
                    color = (0,0,255)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 2:
                    #purple
                    color = (138,43,226)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 3:
                    #green
                    color = (0,128,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 4:
                    #yellow
                    color = (255,255,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 5:
                    #orange
                    color = (255,165,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 6:
                    #pink
                    color = (55,192,203)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 8:
                    #cadet blue
                    color = (95,158,160)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 9:
                    #black
                    color = (255,255,255)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)

            final_images[i,...] = im
            del im

        opts = (
        {
            'title': title, 'width': self.count / 2 * 320,
            'height': self.count / 4 * 240
        })
        self.visualizer.images(final_images.transpose(0,3,1,2), opts=opts,\
            win=self.name + title + "_window")
     
    def show_keypoints(self,images,coordinates,title):
        b, c, h , w = images.size()
        b , keypoints , _ = coordinates.shape
        final_images = numpy.zeros((b,h,w,c))
        radius = 5
        for i, image in enumerate(images):
            #process each image seperately
            im = (image.permute(1,2,0).cpu().numpy().copy() * 255).astype(numpy.int8)
            keypoint = coordinates[i] * h
            for k, key in enumerate(keypoint):
                if k == 0:
                    #red --centroid
                    color = (255,0,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 1:
                    #blue
                    color = (0,0,255)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 2:
                    #purple
                    color = (138,43,226)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 3:
                    #green
                    color = (0,128,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 4:
                    #yellow
                    color = (255,255,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 5:
                    #orange
                    color = (255,165,0)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 6:
                    #pink
                    color = (55,192,203)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 7:
                    #cadet blue
                    color = (95,158,160)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)
                elif k == 8:
                    #black
                    color = (255,255,255)
                    cv2.circle(im, (int(keypoint[k][0]), int(keypoint[k][1])), radius, color,-1)

            final_images[i,...] = im
            del im

        opts = (
        {
            'title': title, 'width': self.count / 2 * 320,
            'height': self.count / 4 * 240
        })
        self.visualizer.images(final_images.transpose(0,3,1,2), opts=opts,\
            win=self.name + title + "_window")

    
    def show_bbox(self,images,coordinates,title):
        b, c, h , w = images.size()
        b , keypoints , _ = coordinates.shape
        final_images = numpy.zeros((b,h,w,c))
        #coordinates ;; b x 8 x 2
        for i, image in enumerate(images):
            #process each image seperately
            im = (image.permute(1,2,0).cpu().numpy().copy() * 255).astype(numpy.int8)
            keypoint = coordinates[i] * h
            for k, key in enumerate(keypoint):
                if (k + 1) % 2 == 0:
                    cv2.line(im, (int(point_1_x), int(point_1_y)), (int(keypoint[k][0]), int(keypoint[k][1])), (0,255,0), 2)
                else:
                    point_1_x = keypoint[k][0]
                    point_1_y = keypoint[k][1]
            final_images[i,...] = im
            del im

        opts = (
        {
            'title': title, 'width': self.count / 2 * 320,
            'height': self.count / 4 * 240
        })
        self.visualizer.images(final_images.transpose(0,3,1,2), opts=opts,\
            win=self.name + title + "_window")

        #v2.line(image, (int(point_1_x), int(point_1_y)), (int(xx), int(yy)), (0,255,0), 1)


def visualisemask(vertices,faces,PdroneToCamera,b,isPredicted = True):
    """Set up renderer for visualising drone based on the predicted pose
        PdroneToCamera: relative pose predicted by network or GT
    """
    if not __KAOLIN_LOADED__:
        return

    device = vertices.get_device()
    vertices = vertices.expand(b,vertices.size(1),vertices.size(2))

    colors = torch.zeros(vertices.size()).to(device)

    
    #setup color
    vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
    vert_max = torch.max(vertices, dim=1, keepdims=True)[0]

    
    if(isPredicted):
        #colors[:,:,:2] = 0
        colors[:,:,:3] = 1
    else:
        colors[:,:,2:3] = 0
        colors[:,:,1] = 1
    
    #get homogeneous coordinates for vertices
    vertices = torch.torch.nn.functional.pad(vertices[:,:,],[0,1],"constant",1.0)
    vertices = vertices.transpose(2,1)

    vertices = torch.matmul(PdroneToCamera,vertices)

    #set up renderer
    renderer = Renderer(320 * 1,240 * 1)
    vertices = vertices.transpose(2,1)

    b , _ ,_ = PdroneToCamera.size()

    #set camera parameters
    cameras = []
    camera_rot_bx3x3 = torch.zeros((b,3,3),dtype=torch.float32).to(device)
    camera_rot_bx3x3[:,0,0] = 1
    camera_rot_bx3x3[:,1,1] = 1
    camera_rot_bx3x3[:,2,2] = 1
    
    cameras.append(camera_rot_bx3x3)
    camera_pos_bx3 = torch.zeros((b,3),dtype=torch.float32).to(device)
    cameras.append(camera_pos_bx3)

    camera_proj_3x1 = torch.zeros((3,1),dtype=torch.float32).to(device)
    camera_proj_3x1[:,:] = torch.from_numpy(geometry.perspectiveprojectionnp(radians(50.8), ratio= 4./3., near=0.3, far=750.0))
    cameras.append(camera_proj_3x1)

    renderer.set_camera_parameters(cameras)

    #convert points from homogeneous
    z_vec =  vertices[..., -1:]
    scale = torch.tensor(1.) / torch.clamp(z_vec, 0.000000001)

    vertices =  vertices[..., :-1]

    #forward pass
    predictions , mask , _ = renderer(points=[vertices,faces.long()],colors_bxpx3=colors)

    return predictions, mask
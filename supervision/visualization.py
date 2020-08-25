import visdom
import numpy
import torch

class NullVisualizer(object):
    def __init__(self):
        self.name = __name__

    def append_loss(self, epoch, global_iteration, loss, mode='train'):
        pass

    def show_images(self, images, title):
        pass

class VisdomVisualizer(object):
    def __init__(self, name, server="http://localhost", count=2):
        print("server",server)
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
                #'legend': mode,
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
            'height': self.count / b * 320
        })
        self.visualizer.images(recon_images, opts=opts,\
            win=self.name + title + "_window")

    def show_activations(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:1, :, :, :]
        maps_cpu = maps_cpu.squeeze(0)
        for i in range(c):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :]
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))

    def show_kernels(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:, :, :, :]
        maps_cpu = maps_cpu.squeeze(0)
        count, _, _ = maps_cpu.size()
        for i in range(count):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :]
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))

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

    def show_point_clouds(self, coords, title):
        point_clouds = coords.detach().cpu()[:self.count, :, :, :]        
        opts = (
        {
            'title': title + '_points3D', 'webgl': True,
            #'legend'=['Predicted', 'Ground Truth'],
            'markersize': 0.5,
            #'markercolor': torch.tensor([[0,0,255], [255,0,0]]).int().numpy(),
            'xtickmin': -3, 'xtickmax': 3, 'xtickstep': 0.2,
            'ytickmin': -3, 'ytickmax': 3, 'ytickstep': 0.2,
            'ztickmin': -2, 'ztickmax': 5, 'ztickstep': 0.2
        })
        for i in range(self.count):
            p3d = point_clouds[i, :, :, :].permute(1, 2, 0).reshape(-1, 3)
            self.visualizer.scatter(X=p3d, opts=opts,\
                win=self.name + "_" + title + '_' + str(i+1))        
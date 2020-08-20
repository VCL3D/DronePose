<h1 id="abstract">Abstract</h1>
 In this work we consider UAVs as cooperative agents supporting human users in their operations. In this context, 
 the 3D localisation of the UAV assistant is an important task that can facilitate the exchange of spatial information between the user and the UAV. 
 To address this in a data-driven manner, we design a data synthesis pipeline to create a realistic multimodal dataset that includes both the exocentric user view,
 and the egocentric UAV view. We then exploit the joint availability of photorealistic and synthesized inputs to train a single-shot monocular pose estimation model.
 During training we leverage diﬀerentiable rendering to supplement a state-of-the-art direct regression objective with a novel smooth silhouette loss.
 Our results demonstrate its qualitative and quantitative performance gains over traditional silhouette objectives.

<h1 id="overview">Overview</h1>

<iframe width="560" height="315" src="https://www.youtube.com/embed/dSbeu238I-I" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<h1>In-The-Wild (YouTube) Results</h1>
<table>
<tr>
<td>
<img src="./assets/images/Outdoor_1.gif" alt="Outdoor_1">
</td>
<td>
<img src="./assets/images/Outdoor_2.gif" alt="Outdoor_2">
</td>
</tr>
<tr>
<td>
<img src="./assets/images/Indoor_1.gif" alt="Indoor_1">
</td>
<td>
<img src="./assets/images/Indoor_2.gif" alt="Indoor_2">
</td>
</tr>
</table>
<h1> Loss Analysis </h1>
We demonstrate that the addition of an exocentric supervision through a differentiable renderer enhances the performace of our method.
However, we further demonstrate that the selection of a smoother loss for the exocentric supervision instead of typical losses (e.g. IoU, GIoU) yields better results and allows us to train a more robust model, able to generalise better.
<h2> Loss Landscape Analysis </h2>
Additionally, we provide a loss landscape analysis and 3D visualization of the loss surfaces.
The IoU has noticeable convexity in contrast to our proposed loss.
<table>
<tr>
<td>
<img src="./assets/images/LossLandscape.png" alt="LossLandscape">
</td>
</tr>
</table>
<h2> Loss distribution </h2>
Then, we present the distribution of the IoU loss compared to the smooth silhouette consistency one across a dense sampling of poses. 
The proposed loss is smoother and contains a better defined minima region.
<table>
<tr>
<td>
<img src="./assets/images/smooth_loss.png" alt="SmoothLossDistribution">
</td>
</tr>
</table>
<h2> Interpolation </h2>
In our first experiment, we selected one random ground-truth pose and we interpolate between it with a random pose.
It can be seen that the proposed loss is more ﬂat than the IoU.
<table>
<tr>
<td>
<img src="./assets/images/InterpolationExp.png" alt="InterpolationExp">
</td>
</tr>
<tr>
<td>
<img src="./assets/images/losses_lerp_1.gif" alt="InterpolationExp_1">
</td>
</tr>
</table>
<h2> Comparison in real data </h2>
Finally, we provide comparison results between a model trained with an IoU loss and another with our proposed smooth objective, in unseen real data. 
The smoothly supervised model minimizes inconsistencies in time and is more robust as it can be seen from the above video.
<table>
<tr>
<td>
<img src="./assets/images/IoUvsSmoothLoss_v1.gif" alt="ComparisonReal">
</td>
<td>
<img src="./assets/images/IoUvsSmoothLoss_v2.gif" alt="ComparisonReal_2">
</td>
</tr>
</table>

 <h1> Data </h1>
The data used to train our methods is a subset of the <a href="https://vcl3d.github.io/UAVA/">UAVA dataset</a>. The data can be downloaded following a two-step proccess.
<h1> Publication </h1>
<p><a href="https://arxiv.org/abs/3330751"><img src="./assets/images/PaperImage.png" title="arXiv paper link" alt="arXiv"></a></p>
<h2> Authors </h2>

<p><a href="https://github.com/tzole1155">Georgios Albanis</a> <strong>*</strong>, <a href="https://github.com/zokin">Nikolaos</a> <a href="https://github.com/zuru">Zioulis</a> <strong>*</strong>, <a href="https://www.iti.gr/iti/people/Anastasios_Dimou.html">Anastasios Dimou</a>, <a href="https://www.iti.gr/iti/people/Dimitrios_Zarpalas.html">Dimitrios Zarpalas</a>, and <a href="https://www.iti.gr/iti/people/Petros_Daras.html">Petros Daras</a></p>

<h2> Citation </h2>


 <h1> Acknowledgements </h1>
 This project has received funding from the European Union’s Horizon 2020 innovation programme [FASTER](https://www.faster-project.eu/) under grant agreement No 833507.

 <table>
<tr>
<td>
<img src="./assets/images/eu.png" alt="eu">
</td>
<td>
<img src="./assets/images/faster.png" alt="faster">
</td>
</tr>
</table>

<!--![eu](./assets/images/eu.png){:width="150px"} ![faster](./assets/images/faster.png){:width="150px"}-->

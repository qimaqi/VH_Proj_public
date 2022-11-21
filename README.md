<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Glanfaloth/TCMR_RELEASE">
    <img src="asset/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Human Pose Estimation from Egocentric Social Interaction Videos</h3>

  <p align="center">
    <br />
    <a href="https://www.youtube.com/playlist?list=PLJKXDihfl_-f0fLAqLN4CImdV7A1XWEf5">View Demo</a>
    ·
    <a href="https://github.com/qimaqi/VH_Proj_public/issues">Report Bug</a>
    ·
    <a href="https://github.com/qimaqi/VH_Proj_public/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The pose estimation with egocentric perspective is important for autonomous robot, augmented reality and health care. However, the dynamic movement, self-occlusion and self-motion in first-person view cause poor performance for the standard pose and shape estimator like Easymocap. In this work we integrate past and future information through PoseForecast module in TCMR. Moreover we design our custom regressor for keypoints estimation and also do extensive ablation study about different Pose Initialization strategy. We achieved amazing performance compared to the YOU2ME original work which formulates camera wearer pose estimation as classification task. Lastly we fit the SMPL model based on estimated keypoints and gain smooth and accurate result compared to running shape estimation directly.

## Conclusion
- We revisited You2Me dataset with “modern methods” as temporal encoding, SMPL body model and channelwise attention methods. Compared to the original YOU2ME method, we aim to predict the interactee body pose with large variance, which was not feasible when YOU2ME was proposed.
- We performed detailed experiments and ablation studies on YOU2ME dataset in regard to network architecture, temporal encoding module, input type as egocentric feature and openpose. We also conducted a detailed analysis towards the prediction error of different
actions.
- We finished the SMPL body mesh prediction pipeline, incorporating the easymocap [1] style tracking and smoothness constrains between frame. Our pipeline gives accurate, smooth and realistic prediction compared to popular repos as Frankmocap [17].

## Limitations and future Work
Due to the limitation of input resolution and lack of camera extrinsic information, we couldn’t perform the SMPL mesh reprojection. It is possible to use more advanced social interation dataset to evaluate the performance of interatee pose estimation and SMPL mesh recover.


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [TCMR]([https://nextjs.org/](https://github.com/hongsukchoi/TCMR_RELEASE))



<!-- USAGE EXAMPLES -->
## Demo

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

![Demo](https://media3.giphy.com/media/eBA3C1GXoIYUSZYxQy/giphy.gif)
![Demo](https://media3.giphy.com/media/wLhMcl5PH6GLFbrhoH/giphy.gif)
![Demo](https://media3.giphy.com/media/0e65JNzfm1pEFYwETo/giphy.gif)
![Demo](https://media3.giphy.com/media/iHTDsDLkvIjwuzPA61/giphy.gif)
![Demo](https://media3.giphy.com/media/s7g6KAuXcC7huCVz57/giphy.gif)
![Demo](https://media3.giphy.com/media/tZ4FcA7IouGdZZs296/giphy.gif)
![Demo](https://media3.giphy.com/media/WBZ0QPcuH3O1VPVGWa/giphy.gif)
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

<p align="right">(<a href="#top">back to top</a>)</p>


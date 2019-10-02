# Pyramid Real Image Denoising Network
This is the code for the paper "Pyramid Real Image Denoising Network". ( VCIP 2019 oral )

[Paper](https://arxiv.org/abs/1908.00273?context=cs.CV)

## 1. Abstract
While deep Convolutional Neural Networks (CNNs) have  shown  extraordinary  capability  of  modelling  specific  noiseand  denoising,  they  still  perform  poorly  on  real-world  noisyimages.  The  main  reason  is  that  the  real-world  noise  is  moresophisticated and diverse. To tackle the issue of blind denoising,in this paper, we propose a novel pyramid real image denoisingnetwork (PRIDNet), which contains three stages. First, the noiseestimation stage uses channel attention mechanism to recalibratethe  channel  importance  of  input  noise.  Second,  at  the  multi-scale  denoising  stage,  pyramid  pooling  is  utilized  to  extractmulti-scale  features.  Third,  the  stage  of  feature  fusion  adopts  akernel selecting operation to adaptively fuse multi-scale features.Experiments  on  two  datasets  of  real  noisy  photographs  demon-strate  that  our  approach  can  achieve  competitive  performancein  comparison  with  state-of-the-art  denoisers  in  terms  of  bothquantitative  measure  and  visual  perception  quality.

## 2. Network Structure

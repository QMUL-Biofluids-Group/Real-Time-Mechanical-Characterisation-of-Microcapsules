## Description of testing data

This folder contains the experimental data of two cases which we have used to test the accuracy and latency of the computational method. The experiments were conducted by Risso et al. (2006), where capsules were flowed through a capillary tube at two flow strengths, leading to two capillary number values Ca=0.02 and 0.052. Note that the capillary number here is defined as $Ca=\mu U/K_s$ where $\mu$ is the viscosity of the channel fluid, $U$ is the average fluid speed and $K_s$ is the membrane area-dilatation modulus. In the experiments of Risso et al. (2006), $K_s$ was measured to be 0.55 N/m with a parallel plate compression method. 

With the images as inputs, the method can predict the membrane law type (within three types) and estimate the values of the membrane shear elastic and area-dilation modulus $G_s$ and $K_s$, respectively. The expected prediction results are shown in the following table. Note that for capsules with the SK and Hooke’s membranes under moderate deformation, it is the membrane elastic parameters instead of the law types that determine the capsule deformation in tube flow. In the table below, the computational latency is the combined image processing and property-prediction time, using a Lenovo laptop with an AMD R7-4800U 1.8G CPU.

|               | Case1_Ca_0.052    | Case2_Ca_0.02 |
| :------------ |:---------------:  | -----:        |
| Law type      | Hooke's law       | SK law        |
| Ks(N/m)       | 0.58              |   0.48        |
| Gs(N/m)       | 0.35              |   0.46        |
| Latency       | 0.96ms            |   0.96ms      |

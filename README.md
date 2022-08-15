# Periodic Orbit Discovery

Periodic orbits are highly desirable trajectories for spacecraft; however they are often difficult to find as a result of highly non-keplerian environments (effects from N-body perturbations, solar radiation pressure, irregular gravity fields, etc). One method to identifying such orbits within these environments requires formulating the problem as a boundary value problem and using numerical shooting methods to iteratively correct initial conditions until periodicity is achieved. This approach, unfortunately, is prone to certain disadvantages. Namely, shooting methods require integrating the equations of motion for the high-fidelity system. These equations require computing high-order gradients of gravitational potential models which can be extremely expensive, particularly if computed numerically. In addition, high-fidelity gravity models are often written in a cartesian description which spans $\mathbb{R}^6$. This domain is particularly large which can make for identifying a compelling converging basin even more difficult.

This work introduces the [Physics-Informed Neural Network Gravity Model (PINN-GM)](https://github.com/joma5012/GravNN) to the periodic orbit discovery problem. The PINN-GM offers a rapidly differentiable and high-accuracy gravity model to be used for integration within traditional shooting methods. Thanks to automatic differentation, the exact gradient and jacobians of the scalar potential can be taken with respect to any coordinate description trivially. 

By using a trained PINN-GM, this work expands the types of shooting method that can be conducted. Namely, no longer does the shooting method need to be conducted in cartesian space. Instead, descriptions like the classical orbital elements can be used in the shooting method. This change considerably reduces the domain of search space due to the natural bounding of the orbital elements (via angle-wrapping or bounded eccentricities) and ensures that the solutions found are not only periodic in cartesian space, but also in element space -- assisting in converging to solutions that maintain desireable orbital characteristics over extended periods of time. Moreover, by using orbital element shooting methods, constrained optimizations can be used which enforce bounds on particular orbital elements when searching for a periodic orbit solution which can assist trajectory designers in searching for stable trajectories which also satisfy particular mission requirements.

<div align="center">
  <img src="docs/source/assets/figure_integrated_IC_cart_1_corrected_x10.pdf">
</div>

<div align="center">
  <img src="docs/source/assets/figure_integrated_IC_1_corrected_x10.pdf ">
</div>


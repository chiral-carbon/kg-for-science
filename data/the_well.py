all_datasets = [
    "MHD_256",
    "MHD_64",
    "acoustic_scattering_discontinuous_2d",
    "acoustic_scattering_inclusions_2d",
    "acoustic_scattering_maze_2d",
    "active_matter",
    "convective_envelope_rsg",
    "euler_quadrants",
    "helmholtz_staircase",
    "pattern_formation",
    "planetswe",
    "post_neutron_star_merger",
    "rayleigh_benard",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_128",
    "supernova_explosion_64",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D",
    "viscoelastic_instability",
]

full_string = """
>> Magnetohydrodynamics (MHD) compressible turbulence

**One line description of the data:** This is an MHD fluid flows in the compressible limit (subsonic, supersonic, sub-Alfvenic, super-Alfvenic).

**Longer description of the data:** An essential component of the solar wind, galaxy formation, and of interstellar medium (ISM) dynamics is magnetohydrodynamic (MHD) turbulence. This dataset consists of isothermal MHD simulations without self-gravity (such as found in the diffuse ISM) initially generated with resolution $256^3$ and then downsampled to $64^3$ after anti-aliasing with an ideal low-pass filter.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/abc484/pdf)

**Domain expert**: [Blakesley Burkhart](https://www.bburkhart.com/), CCA, Flatiron Institute & Rutgers University.

**Code or software used to generate the data**: Fortran + MPI.

**Equation**: 
```math
\begin{align}
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) &= 0 \nonumber\\
\frac{\partial \rho \mathbf{v}}{\partial t} + \nabla \cdot (\rho \mathbf{v} \mathbf{v} - \mathbf{B} \mathbf{B}) + \nabla p &= 0 \nonumber\\
\frac{\partial \mathbf{B}}{\partial t} - \nabla \times (\mathbf{v} \times \mathbf{B}) &= 0 \nonumber\\
\end{align}
```
where $\rho$ is the density, $\mathbf{v}$ is the velocity, $\mathbf{B}$ is the magnetic field, $\mathbf{I}$ the identity matrix and $p$ is the gas pressure.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/MHD_256/gif/density_normalized.gif)


> About the data

**Dimension of discretized data:** 100 timesteps of $256\times 256\times256$ cubes.

**Fields available in the data:** Density (scalar field), velocity (vector field), magnetic field (vector field).

**Number of trajectories:** 10 Initial conditions x 10 combination of parameters = 100 trajectories.

**Estimated size of the ensemble of all simulations:** 4.58TB.

**Grid type:** uniform grid, cartesian coordinates.

**Initial conditions:** uniform IC.

**Boundary conditions:** periodic boundary conditions.

**Data are stored separated by ($\Delta t$):** 0.01 (arbitrary units).

**Total time range ($t\_{min}$ to $t\_{max}$):** $t\_{min} = 0$, $t\_{max} = 1$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** dimensionless so 256 pixels.

**Set of coefficients or non-dimensional parameters evaluated:** all combinations of $\mathcal{M}_s=${0.5, 0.7, 1.5, 2.0 7.0} and $\mathcal{M}_A =${0.7, 2.0}.

**Approximate time to generate the data:** 48 hours per simulation.

**Hardware used to generate the data**: 64 cores.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** MHD fluid flows in the compressible limit (sub and super sonic, sub and super Alfvenic).

**How to evaluate a new simulator operating in this space:** Check metrics such as Power spectrum, two points correlation function.



>> Magnetohydrodynamics (MHD) compressible turbulence

**One line description of the data:** This is an MHD fluid flows in the compressible limit (subsonic, supersonic, sub-Alfvenic, super-Alfvenic). 

**Longer description of the data:** An essential component of the solar wind, galaxy formation, and of interstellar medium (ISM) dynamics is magnetohydrodynamic (MHD) turbulence. This dataset consists of isothermal MHD simulations without self-gravity (such as found in the diffuse ISM) initially generated with resolution $256^3$ and then downsampled to $64^3$ after anti-aliasing with an ideal low-pass filter. This dataset is the downsampled version.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/abc484/pdf)

**Domain expert**: [Blakesley Burkhart](https://www.bburkhart.com/), CCA, Flatiron Institute & Rutgers University.

**Code or software used to generate the data**: Fortran + MPI.

**Equation**: 
```math
\begin{align}
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) &= 0 \nonumber\\
\frac{\partial \rho \mathbf{v}}{\partial t} + \nabla \cdot (\rho \mathbf{v} \mathbf{v} - \mathbf{B} \mathbf{B}) + \nabla p &= 0 \nonumber\\
\frac{\partial \mathbf{B}}{\partial t} - \nabla \times (\mathbf{v} \times \mathbf{B}) &= 0 \nonumber\\
\end{align}
```
where $\rho$ is the density, $\mathbf{v}$ is the velocity, $\mathbf{B}$ is the magnetic field, $\mathbf{I}$ the identity matrix and $p$ is the gas pressure.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/MHD_64/gif/density_unnormalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| MHD_64  | 0.351 | 0.314 |0.270|$\mathbf{0.211}$|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** 100 timesteps of $64\times 64\times 64$ cubes.

**Fields available in the data:** Density (scalar field), velocity (vector field), magnetic field (vector field).

**Number of trajectories:** 10 Initial conditions x 10 combination of parameters = 100 trajectories.

**Estimated size of the ensemble of all simulations:** 71.6 GB.

**Grid type:** uniform grid, cartesian coordinates.

**Initial conditions:** uniform IC.

**Boundary conditions:** periodic boundary conditions.

**Data are stored separated by ($\Delta t$):** 0.01 (arbitrary units).

**Total time range ($t\_{min}$ to $t\_{max}$):** $t\_{min} = 0$, $t\_{max} = 1$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** dimensionless so 64 pixels.

**Set of coefficients or non-dimensional parameters evaluated:** all combinations of $\mathcal{M}_s=${0.5, 0.7, 1.5, 2.0 7.0} and $\mathcal{M}_A =${0.7, 2.0}.

**Approximate time to generate the data:** Downsampled from MHD_256 after applying ideal low-pass filter.

**Hardware used to generate the data**: Downsampled from MHD_256 after applying ideal low-pass filter.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** MHD fluid flows in the compressible limit (sub and super sonic, sub and super Alfvenic).

**How to evaluate a new simulator operating in this space:** Check metrics such as Power spectrum, two-points correlation function.

>> Acoustic Scattering - Single Discontinuity

**One line description of the data:** Simple acoustic wave propogation over a domain split into two continuously varying sub-domains with a single discountinuous interface. 

**Longer description of the data:** These variable-coefficient acoustic equations describe the propogation of an acoustic pressure wave through domains consisting of multiple materials with different scattering properties. This problem emerges in source optimization and it's inverse - that of identifying the material properties from the scattering of the wave - is a vital problem in geology and radar design. This is the simplest of three scenarios. In this case, we have a variable number of initial point sources and single discontinuity separating two sub-domains. Within each subdomain, the density of the underlying material varies smoothly. 

**Domain expert**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: Clawpack, adapted from: http://www.clawpack.org/gallery/pyclaw/gallery/acoustics_2d_interface.html

**Equation**:

```math
\begin{align}
\frac{ \partial p}{\partial t} + K(x, y) \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) &= 0 \\
\frac{ \partial u  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial x} &= 0 \\
\frac{ \partial v  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial v} &= 0 
\end{align}
```

with $\rho$ the material density, $u, v$ the velocity in the $x, y$ directions respectively, $p$ the pressure, and $K$ the bulk modulus. 

Example material densities can be seen below:

![image](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/acoustic_scattering_discontinuous_2d/gif/discontinuous_density.png)

> About the data

**Dimension of discretized data:** $101$ steps of $256\times 256$ images.

**Fields available in the data:** pressure (scalar field), material density (constant scalar field), material speed of sound (constant scalar field), velocity field (vector field). 

**Number of trajectories:** 2000.

**Estimated size of the ensemble of all simulations:** 157.7 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Flat pressure static field with 1-4 high pressure rings randomly placed in domain. The rings are defined with variable intensity $\sim \mathcal U(.5, 2)$ and radius $\sim \mathcal U(.06, .15)$. 

**Boundary conditions:** Open domain in $y$, reflective walls in $x$.

**Simulation time-step:** Variable based on CFL with safety factor .25. 

**Data are stored separated by ($\Delta t$):** 2/101. 

**Total time range ($t_{min}$ to $t_{max}$):** [0, 2.]

**Spatial domain size ($L_x$, $L_y$, $L_z$):** [-1, 1] x [-1, 1]

**Set of coefficients or non-dimensional parameters evaluated:**

$K$ is fixed at 4.0. 

$\rho$ is the primary coefficient here. Each side is generated with one of the following distributions:
- Gaussian Bump - Peak density samples from $\sim\mathcal U(1, 7)$ and $\sigma \sim\mathcal U(.1, 5)$ with the center of the bump uniformly sampled from the extent of the subdomain.
- Linear gradient - Four corners sampled with $\rho \sim \mathcal U(1, 7)$. Inner density is bilinearly interpolated.
- Constant - Constant $\rho \sim\mathcal U(1, 7)$
- Smoothed Gaussian Noise - Constant background sampled $\rho \sim\mathcal U(1, 7)$ with IID standard normal noise applied. This is then smoothed by a Gaussian filter of varying sigma $\sigma \sim\mathcal U(5, 10)$

**Approximate time to generate the data:** ~15 minutes per simulation. 

**Hardware used to generate the data and precision used for generating the data:** 64 Intel Icelake cores per simulation. Generated in double precision.

> What is interesting and challenging about the data:
Wave propogation through discontinuous media. Most existing machine learning datasets for computational physics are highly smooth and the acoustic challenges presented here offer challenging discontinuous scenarios that approximate complicated geometry through the variable density.

>> Acoustic Scattering - Inclusions

**One line description of the data:** Simple acoustic wave propogation over a domain split into two continuously varying sub-domains with a single discountinuous interface. With additive randomly generating inclusions (materials of significantly different density). 

**Longer description of the data:** These variable-coefficient acoustic equations describe the propogation of an acoustic pressure wave through domains consisting of multiple materials with different scattering properties. This problem emerges in source optimization and it's inverse - that of identifying the material properties from the scattering of the wave - is a vital problem in geology and radar design. In this case, we have a variable number of initial point sources and a domain with random inclusions. These types of problems are of particular interest in geology where the inverse scattering is used to identify mineral deposits. 

**Domain expert**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: Clawpack, adapted from: http://www.clawpack.org/gallery/pyclaw/gallery/acoustics_2d_interface.html

**Equation**:

```math
\begin{align}
\frac{ \partial p}{\partial t} + K(x, y) \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) &= 0 \\
\frac{ \partial u  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial x} &= 0 \\
\frac{ \partial v  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial v} &= 0 
\end{align}
```
with $\rho$ the material density, $u, v$ the velocity in the $x, y$ directions respectively, $p$ the pressure, and $K$ the bulk modulus. 

Example material densities can be seen below:

![image](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/acoustic_scattering_inclusions_2d/gif/inclusions_density.png)

> About the data

**Dimension of discretized data:** $101$ steps of $256\times256$ images.

**Fields available in the data:** pressure (scalar field), material density (constant scalar field), material speed of sound (constant scalar field), velocity field (vector field).

**Number of trajectories:** 4000.

**Estimated size of the ensemble of all simulations:** 283.8 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Flat pressure static field with 1-4 high pressure rings randomly placed in domain. The rings are defined with variable intensity $\sim \mathcal U(.5, 2)$ and radius $\sim \mathcal U(.06, .15)$. 

**Boundary conditions:** Open domain in $y$, reflective walls in $x$.

**Simulation time-step:** Variable based on CFL with safety factor .25. 

**Data are stored separated by ($\Delta t$):** 2/101. 

**Total time range ($t_{min}$ to $t_{max}$):** [0, 2.].

**Spatial domain size ($L_x$, $L_y$, $L_z$):** [-1, 1] x [-1, 1].

**Set of coefficients or non-dimensional parameters evaluated:**

$K$ is fixed at 4.0. 

$\rho$ is the primary coefficient here. This is a superset of the single discontinuity example so the background is first generated two splits with one of the following distributions:
- Gaussian Bump - Peak density samples from $\sim\mathcal U(1, 7)$ and $\sigma \sim\mathcal U(.1, 5)$ with the center of the bump uniformly sampled from the extent of the subdomain.
- Linear gradient - Four corners sampled with $\rho \sim \mathcal U(1, 7)$. Inner density is bilinearly interpolated.
- Constant - Constant $\rho \sim\mathcal U(1, 7)$
- Smoothed Gaussian Noise - Constant background sampled $\rho \sim\mathcal U(1, 7)$ with IID standard normal noise applied. This is then smoothed by a Gaussian filter of varying sigma $\sigma \sim\mathcal U(5, 10)$. 

Inclusions are then added as 1-15 random ellipsoids with center uniformly sampled from the domain and height/width sampled uniformly from [.05, .6]. The ellipsoid is then rotated randomly with angle sampled [-45, 45]. For the inclusions, $Ln(\rho)\sim \mathcal U(-1, 10)$ 


**Approximate time to generate the data:** ~15 minutes per simulation. 

**Hardware used to generate the data and precision used for generating the data:** 64 Intel Icelake cores per simulation. Generated in double precision.

> What is interesting and challenging about the data:

Wave propogation through discontinuous media. Most existing machine learning datasets for computational physics are highly smooth and the acoustic challenges presented here offer challenging discontinuous scenarios that approximate complicated geometry through the variable density. The inclusions change wave propogation speed but only in small, irregular areas.

>> Acoustic Scattering - Maze

**One line description of the data:** Simple acoustic wave propogation through maze-like structures.

**Longer description of the data:** These variable-coefficient acoustic equations describe the propogation of an acoustic pressure wave through maze-like domains. Pressure waves emerge from point sources and propogate through domains consisting of low density maze paths and orders of magnitude higher density maze walls. This is built primarily as a challenge for machine learning methods, though has similar properties to optimal placement problems like WiFi in a building. 

**Domain expert**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: Clawpack, adapted from: http://www.clawpack.org/gallery/pyclaw/gallery/acoustics_2d_interface.html

**Equation**:

```math
\begin{align}
\frac{ \partial p}{\partial t} + K(x, y) \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) &= 0 \\
\frac{ \partial u  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial x} &= 0 \\
\frac{ \partial v  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial v} &= 0 
\end{align}
```
with $\rho$ the material density, $u, v$ the velocity in the $x, y$ directions respectively, $p$ the pressure, and $K$ the bulk modulus. 

Example material densities can be seen below:

![image](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/acoustic_scattering_maze_2d/gif/mazes_density.png)

Traversal can be seen:

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/acoustic_scattering_maze_2d/gif/pressure_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| acoustic_scattering_maze  | 1.00 | 1.00| 1.00| $\mathbf{0.173}$|

> About the data

**Dimension of discretized data:** $201$ steps of $256\times256$ images.

**Fields available in the data:** pressure (scalar field), material density (constant scalar field), material speed of sound (constant scalar field), velocity field (vector field).
**Number of trajectories:** 2000.

**Estimated size of the ensemble of all simulations:** 311.3 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Flat pressure static field with 1-6 high pressure rings randomly placed along paths of maze. The rings are defined with variable intensity $\sim \mathcal U(3., 5.)$ and radius $\sim \mathcal U(.01, .04)$. Any overlap with walls is removed. 

**Boundary conditions:** Open domain in $y$, reflective walls in $x$.

**Simulation time-step:** Variable based on CFL with safety factor .25. 

**Data are stored separated by ($\Delta t$):** 2/201. 

**Total time range ($t_{min}$ to $t_{max}$):** [0,4.].

**Spatial domain size ($L_x$, $L_y$, $L_z$):** [-1, 1] x [-1, 1].

**Set of coefficients or non-dimensional parameters evaluated:**

$K$ is fixed at 4.0. 

$\rho$ is the primary coefficient here. We generated a maze with initial width between 6 and 16 pixels and upsample it via nearest neighbor resampling to create a 256 x 256 maze. The walls are set to $\rho=10^6$ while paths are set to  $\rho=3$.  

**Approximate time to generate the data:** ~20 minutes per simulation. 

**Hardware used to generate the data and precision used for generating the data:** 64 Intel Icelake cores per simulation. Generated in double precision.

> What is interesting and challenging about the data:

This is an example of simple dynamics in complicated geometry. The sharp discontinuities can be a significant problem for machine learning models, yet they are a common feature in many real-world physics. While visually the walls appear to stop the signal, it is actually simply the case that the speed of sound is much much lower inside the walls leading to partial reflection/absorbtion at the interfaces.

>> Active fluid simulations

**One line description of the data:**  Modeling and simulation of biological active matter.

**Longer description of the data:** Simulation of a continuum theory describing the dynamics of $N$ rod-like active particles immersed in a Stokes fluid having linear dimension $L$ and colume $L^2$.

**Associated paper**: https://arxiv.org/abs/2308.06675

**Domain expert**: [Suryanarayana Maddu](https://sbalzarini-lab.org/?q=alumni/surya), CCB, Flatiron Institute. 

**Code or software used to generate the data**: [Github repository](https://github.com/SuryanarayanaMK/Learning_closures/tree/master)

**Equations**: Equations (1) to (5) of the associated paper.


![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/active_matter/gif/concentration_notnormalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| active_matter  | $\mathbf{0.982}$  | 143 |2.77|1.09|

Preliminary benchmarking, in VRMSE.





> About the data

**Dimension of discretized data:** $81$ time-steps of $256\times256$ images per trajectory.

**Fields available in the data:** concentration (scalar field),
velocity (vector field), orientation tensor (tensor field), strain-rate tensor (tensor field).


**Number of trajectories:** $5$ trajectories per parameter-set, each trajectory being generated with a different initialization of the state field {$c,D,U$}.

**Size of the ensemble of all simulations:** 51.3 GB.

**Grid type:** Uniform grid, cartesian coordinates.

**Initial conditions:** The concentration is set to constant value $c(x,t)=1$ and the orientation tensor is initialized as plane-wave perturbation about the isotropic state.

**Boundary conditions:** Periodic boundary conditions.

**Simulation time-step:** $3.90625\times 10^{-4}$ seconds.

**Data are stored separated by ($\Delta t$):** 0.25 seconds.

**Total time range ($t_{min}$ to $t_{max}$):** $0$ to $20$ seconds.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $L_x=10$ and $L_y=10$

**Set of coefficients or non-dimensional parameters evaluated:** $\alpha =$ {-1,-2,-3,-4,-5}; $\beta  =$ {0.8}; 
$\zeta =$ {1,3,5,7,9,11,13,15,17}.

**Approximate time and hardware to generate the data:** 20 minutes per simulation on an A100 GPU in double precision. There is a total of 225 simulations, which is approximately 75 hours.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** How is energy being transferred between scales? How is vorticity coupled to the orientation field? Where does the transition from isotropic state to nematic state occur with the change in alignment ($\zeta$) or dipole strength ($\alpha$)? 


**How to evaluate a new simulator operating in this space:** Reproducing some summary statistics like power spectra and average scalar order parameters. Additionally, being able to accurately capture the phase transition from isotropic to nematic state.


>> Red Supergiant Convective Envelope

**One line description of the data:** 3D radiation hydrodynamic simulations of convective envelopes of red supergiant stars.

**Longer description of the data:** Massive stars evolve into red supergiants, which have large radii and luminosities, and low-density, turbulent, convective envelopes. These simulations model the (inherently 3D) convective properties and gives insight into the progenitors of supernovae explosions.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac5ab3)

**Domain experts**: [Yan-Fei Jiang](https://jiangyanfei1986.wixsite.com/yanfei-homepage) (CCA, Flatiron Institute), [Jared Goldberg](https://jaredagoldberg.wordpress.com/) (CCA, Flatiron Institute), [Jeff Shen](https://jshen.net) (Princeton University).

**Code or software used to generate the data**: [Athena++](https://www.athena-astro.app/)

**Equations**
```math
\begin{align*}
\frac{\partial\rho}{\partial t}+\mathbf{\nabla}\cdot(\rho\mathbf{v})&=0\\
\frac{\partial(\rho\mathbf{v})}{\partial t}+\mathbf{\nabla}\cdot({\rho\mathbf{v}\mathbf{v}+{{\sf P_{\rm gas}}}}) &=-\mathbf{G}_r-\rho\mathbf{\nabla}\Phi \\
\frac{\partial{E}}{\partial t}+\mathbf{\nabla}\cdot\left[(E+ P_{\rm gas})\mathbf{v}\right] &= -c G^0_r -\rho\mathbf{v}\cdot\mathbf{\nabla}\Phi\\
\frac{\partial I}{\partial t}+c\mathbf{n}\cdot\mathbf{\nabla} I &= S(I,\mathbf{n})
\end{align*}
```
where 
- $\rho$ = gas density
- $\mathbf{v}$ = flow velocity
- ${\sf P_{\rm gas}}$ = gas pressure tensor
- $P_{\rm gas}$ = gas pressure scalar
- $E$ = total gas energy density
    - $E = E_g + \rho v^2 / 2$, where $E_g = 3 P_{\rm gas} / 2$ = gas internal energy density
- $G^0_r$ and $\mathbf{G}_r$ = time-like and space-like components of the radiation four-force
- $I$ = frequency integrated intensity, which is a function of time, spatial coordinate, and photon propagation direction $\mathbf{n}$
- $\mathbf{n}$ = photon propagation direction

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/convective_envelope_rsg/gif/density_unnormalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| convective_envelope_rsg  | 1.08  | $\mathbf{1.06}$ |1.15|1.07|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** $100$ time-steps of $256\times 128 \times 256$ images per trajectory.

**Fields available in the data:** energy (scalar field), density (scalar field), pressure (Scalar field), velocity (vector field).

**Number of trajectories:** 29 (they are cuts of one long trajectory, long trajectory available on demand).

**Estimated size of the ensemble of all simulations:** 570 GB.

**Grid type:** spherical coordinates, uniform in $(\log r, \theta,\phi)$.  Simulations are done for a portion of a sphere (not the whole sphere), so the simulation volume is like a spherical cake slice.

**Initial and boundary conditions:** The temperature at the inner boundary (IB) is first set to equal that of the appropriate radius coordinate in the MESA (1D) model ($400\~R_\odot$ and $300\~R_\odot$) and the density selected to approximately recover the initial total mass of the star in the simulation ($15.4\~M_\odot$ and $14\~M_\odot$). 
Between $300\~R_\odot$ and $400\~R_\odot$, the initial profile is constructed with the radiative luminosity to be $10^5\~L_\odot$, and this is kept fixed in the IB.


**Simulation time-step:** 198 days.

**Data are stored separated by ($\Delta t$):** units here are sort of arbitrary, $\Delta t= 8$.

**Total time range ($t_{min}$ to $t_{max}$):**  0, 806 (arbitrary).

**Spatial domain size:** $R$ from $300-6700~{\rm R_\odot}$, θ from $π/4−3π/4$ and $\phi$ from $0−π$, with $δr/r ≈ 0.01$.

**Set of coefficients or non-dimensional parameters evaluated:**

| Simulation | radius of inner boundary $R_{IB}/R_\odot$ | radius of outer boundary $R_{OB}/R_\odot$ | heat source | resolution (r × θ × $\phi$) | duration | core mass $mc/M\odot$ | final mass $M_{\rm final}/M_\odot$ |
|--|--|--|--|--|--|--|--|
| Whole simulation (to obtain the 29 trajectories) | 300 | 6700 | fixed L | 256 × 128 × 256 | 5766 days | 10.79 | 12.9 |

**Approximate time to generate the data:** 2 months on 80 nodes for each run.

**Hardware used to generate the data:** 80x NASA Pleiades Skylake CPU nodes.

> What is interesting and challenging about the data:

**What phenomena of physical interest are captured in the data:** turbulence and convection (inherently 3D processes), variability.

**How to evaluate a new simulator operating in this space:** can it predict behaviour of simulation in convective steady-state, given only perhaps a few snapshots at the beginning of the simulation?

**Caveats:** complicated geometry, size of a slice in R varies with R (think of this as a slice of cake, where the parts of the slice closer to the outside have more area/volume than the inner parts), simulation reaches convective steady-state at some point and no longer "evolves".

>> Euler Multi-quadrants - Riemann problems (compressible, inviscid fluid)

**One line description of the data:**  Evolution of different gases starting with piecewise constant initial data in quadrants.

**Longer description of the data:**  The evolution can give rise to shocks, rarefaction waves, contact discontinuities, interaction with each other and domain walls.

**Associated paper**: [Paper](https://epubs.siam.org/doi/pdf/10.1137/S1064827595291819?casa_token=vkASCwD4WngAAAAA:N0jy0Z6tshitF10_YRTlZzU-P7mAiPFr3v58sw7pmRsZOarAi824-b1CWhOQts1rvaG3YpJisw)

**Domain experts**: [Marsha Berger](https://cs.nyu.edu/~berger/)(Flatiron Institute & NYU), [Ruben Ohana](https://rubenohana.github.io/) (CCM, Flatiron Institute & Polymathic AI), [Michael McCabe](https://mikemccabe210.github.io/) (Polymathic AI).

**Code or software used to generate the data**: [Clawpack (AMRClaw)](http://www.clawpack.org/)

**Equation**: Euler equations for a compressible gas:
```math
\begin{align}
U_t + F(U)_x + G(U)_y &= 0 \nonumber\\
\textrm{where} \quad U = \begin{bmatrix} \rho \nonumber\\ \rho u \\ \rho v \\ e \end{bmatrix}, \quad F(U) = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(e + p) \end{bmatrix},& \quad G(U) = \begin{bmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(e + p) \end{bmatrix}, \quad \\ e = \frac{p}{(\gamma - 1)} + \frac{\rho (u^2 + v^2)}{2}&, \quad p = A\rho^{\gamma}. \nonumber
\end{align}
```
with $\rho$ the density, $u$ and $v$ the $x$ and $y$ velocity components, $e$ the energy, $p$ the pressure, $\gamma$ the gas constant, and $A>0$ is a function of entropy.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/euler_quadrants/gif/density_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| euler_multi-quadrants (periodic b.c. only)  | 2.22  | 2.19 |$\mathbf{1.98}$ |2.26|

> About the data

**Dimension of discretized data:** 100 timesteps of 512x512 images.

**Fields available in the data:** density (scalar field), energy (scalar field), pressure (scalar field), momentum (vector field).

**Number of trajectories:** 500 per set of parameters, 10 000 in total.

**Estimated size of the ensemble of all simulations:** 5.17 TB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Randomly generated initial quadrants.

**Boundary conditions:** Periodic or open.

**Simulation time-step:** variable.

**Data are stored separated by ($\Delta t$):** 0.015s (1.5s for 100 timesteps).

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max}=1.5s$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $L_x = 1$ and  $L_y = 1$.

S**et of coefficients or non-dimensional parameters evaluated:** all combinations of $\gamma$ constant of the gas: $\gamma=${1.3,1.4,1.13,1.22,1.33,1.76, 1.365,1.404,1.453,1.597} and boundary conditions: {extrap, periodic}.

**Approximate time to generate the data:** 80 hours on 160 CPU cores for all data.

**Hardware used to generate the data and precision used for generating the data:** Icelake nodes, double precision.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** capture the shock formations and interactions. Multiscale shocks.

**How to evaluate a new simulator operating in this space:** the new simulator should predict the shock at the right location and time, and the right shock strength, as compared to a ‘pressure’ gauge monitoring the ‘exact’ solution.

>> Helmholtz equation on a 2D staircase

**One line description of the data:** First high-order accurate solution of
acoustic scattering from a nonperiodic source by a periodic surface, relevant for its use in waveguide applications (antennae, diffraction from gratings, photonic/phononic crystals, noise cancellation, seismic filtering, etc.).

**Longer description of the data:**  Accurate solution of PDEs near infinite, periodic boundaries poses a numerical challenge due these surfaces serving as waveguides, allowing modes to propagate for long distances from the source. This property makes numerical truncation of the (infinite) solution domain unfeasible, as it would induce large artificial reflections and therefore errors. Periodization (reducing the computational domain to one unit cell) is only possible if the incident wave is also
periodic, such as plane waves, but not for nonperiodic sources, e.g. a point source. Computing a high-order accurate scattering solution from a point source, however, would be of scientific interest as it models applications such as remote sensing, diffraction from gratings, antennae, or acoustic/photonic metamaterials. We use a combination of the Floquet—Bloch transform (also known as array scanning method) and boundary integral equation methods to alleviate these challenges and recover the scattered solution as an integral over a family of quasiperiodic solutions parameterized by their on-surface wavenumber. The advantage of this approach is that each of the quasiperiodic solutions may be computed quickly by periodization, and accurately via high-order quadrature.

**Associated paper**: https://arxiv.org/abs/2310.12486

**Domain expert**: [Fruzsina Julia Agocs](https://fruzsinaagocs.github.io/), CCM, Flatiron Institute.

**Code or software used to generate the data**: [Github repository](https://www.github.com/fruzsinaagocs/bies)

**Equations**:

While we solve equations in the frequency domain, the original time-domain problem is:

$$\frac{\partial^2 U(t, \mathbf{x})}{\partial t^2} - \Delta U(t, \mathbf{x}) = \delta(t)\delta(\mathbf{x} - \mathbf{x}_0), $$
where $\Delta = \nabla \cdot \nabla$ is the spatial Laplacian. [ADD what is U]
The sound-hard boundary $\partial \Omega$ imposes Neumann boundary conditions,

$$ U_n(t, \mathbf{x}) = \mathbf{n} \cdot \nabla U = 0, \quad t \in \mathbb{R}, \quad \mathbf{x} \in \partial \Omega. $$

Upon taking the temporal Fourier transform, we get the inhomogeneous Helmholtz Neumann boundary value problem
```math
\begin{align}
-(\Delta + \omega^2)u &= \delta_{\mathbf{x}_0}, \quad \text{in } \Omega,\\
u_n &= 0 \quad \text{on } \partial \Omega,
\end{align}
```
with outwards radiation conditions as described in [1]. The region $\Omega$ lies above a corrugated boundary $\partial \Omega$, extending with spatial period $d$ in the $x_1$ direction, and is unbounded in the positive $x_2$ direction. The current example is a right-angled staircase whose unit cell consists of two equal-length line segments at $\pi/2$ angle to each other.

<div style="transform: rotate(90deg);">
  <img src="https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/helmholtz_staircase/gif/pressure_re_normalized.gif" alt="Rotated GIF">
</div>

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| helmholtz_staircase  |0.00555 |$\mathbf{0.00205}$ | 0.0825 | 0.00520|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** $50$ time-steps of 
$1024\times256$ images.

**Fields available in the data:**
real and imaginary part of accoustic pressure (scalar field), the staircase mask (scalar field, stationary).

**Number of trajectories:** $512$ (combinations of $16$ input parameter $\omega$ and $32$ source positions $\mathbf{x}_0$).

**Size of the ensemble of all simulations:** 52.4 GB.

**Grid type:** uniform.

**Initial conditions:** The time-dependence is
analytic in this case: $U(t, \mathbf{x}) = u(\mathbf{x})e^{-i\omega t}.$ Therefore any spatial solution may serve as an initial condition.

**Boundary conditions:** Neumann conditions (normal
derivative of the pressure $u$ vanishes, with the normal defined as pointing up from
the boundary) are enforced at the boundary.

**Simulation time-step:** continuous in time (time-dependence is
analytic).

**Data are stored separated by ($\Delta t$):** $\Delta t =\frac{2\pi}{\omega N}$, with $N = 50$.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{\mathrm{min}} = 0$, $t_{\mathrm{max}} =
\frac{2\pi}{\omega}$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $-8.0 \leq x_1 \leq 8.0$ horizontally, and $-0.5 \geq x_2 \geq 3.5$ vertically.

**Set of coefficients or non-dimensional parameters evaluated:** $\omega$={0.06283032, 0.25123038, 0.43929689, 0.62675846, 0.81330465, 0.99856671, 1.18207893, 1.36324313, 1.5412579, 1.71501267, 1.88295798, 2.04282969, 2.19133479, 2.32367294, 2.4331094,  2.5110908}, with the sources coordinates being all combinations of $x$={-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4} and $y$={-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4}.

**Approximate time to generate the data:** per input parameter: $\sim 400s$, total: $\sim 50$ hours.

**Hardware used to generate the data:** 64 CPU cores.

> What is interesting and challenging about the data:

**What phenomena of physical interest are captured in the data:** The simulations capture the existence of trapped acoustic waves – modes that are guided along the corrugated surface. They also show that the on-surface wavenumber of trapped modes is different than the frequency of the incident radiation, i.e. they capture the trapped modes’ dispersion relation.

**How to evaluate a new simulator operating in this space:**
The (spatial) accuracy of a new simulator/method could be checked by requiring that it conserves flux – whatever the source injects into the system also needs to come out. The trapped modes’ dispersion relation may be another metric, my method generates this to 7-8 digits of accuracy at the moment, but 10-12 digits may also be obtained. The time-dependence learnt by a machine learning algorithm can be compared to the analytic solution $e^{-i\omega t}$, this can be used to evaluate temporal accuracy.

>> Pattern formation in the Gray-Scott reaciton-diffusion equations

**One line description of the data:** Stable Turing patterns emerge from randomness, with drastic qualitative differences in pattern dynamics depending on the equation parameters.

**Longer description of the data:** The Gray-Scott equations are a set of coupled reaction-diffusion equations describing two chemical species, $A$ and $B$, whose concentrations vary in space and time. The two parameters $f$ and $k$ control the “feed” and “kill” rates in the reaction. A zoo of qualitatively different static and dynamic patterns in the solutions are possible depending on these two parameters. There is a rich landscape of pattern formation hidden in these equations. 

**Associated paper**: None.

**Domain expert**: [Daniel Fortunato](https://danfortunato.com/), CCM and CCB, Flatiron Institute.

**Code or software used to generate the data**: [Github repository](https://github.com/danfortunato/spectral-gray-scott) (MATLAB R2023a, using the stiff PDE integrator implemented in Chebfun. The Fourier spectral method is used in space (with nonlinear terms evaluated pseudospectrally), and the exponential time-differencing fourth-order Runge-Kutta scheme (ETDRK4) is used in time.)

**Equation describing the data** 
```math
\begin{align}
\frac{\partial A}{\partial t} &= \delta_A\Delta A - AB^2 + f(1-A) \nonumber \\
\frac{\partial B}{\partial t} &= \delta_B\Delta B - AB^2 + (f+k)B \nonumber
\end{align}
```
The dimensionless parameters describing the behavior are: $f$, $k$, $\frac{\delta_A}{\delta_B}$


![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/pattern_formation/gif/concentration_A_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| pattern_formation | 0.178  | 0.177 | 0.322|$\mathbf{0.0512}$|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** 1001 time-steps of $128\times 128$ images.

**Fields available in the data:** Two chemical species $A$ and $B$.

**Number of trajectories:** 6 sets of parameters, 200 initial conditions per set = 1200.

**Estimated size of the ensemble of all simulations:** 153.8 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Two types of initial conditions generated: random Fourier series and random clusters of Gaussians.

**Boundary conditions:** periodic.

**Simulation time-step:** 1 second.

**Data are stored separated by ($\Delta t$):** 10 seconds.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} =0$, $t_{max} = 10,000$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $x,y\in[-1,1]$.

**Set of coefficients or non-dimensional parameters evaluated:** All simulations used $\delta_u = 2.10^{-5}$ and $\delta_v = 1.10^{-5}$.
"Gliders": $f = 0.014, k = 0.054$. "Bubbles": $f = 0.098, k =0.057$. "Maze": $f= 0.029, k = 0.057$. "Worms": $f= 0.058, k = 0.065$. "Spirals": $f=0.018, k = 0.051$. "Spots": $f= 0.03, k=0.062$.

**Approximate time to generate the data:** 5.5 hours per set of parameters, 33 hours total.

**Hardware used to generate the data:** 40 CPU cores.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** Pattern formation: by sweeping the two parameters $f$ and $k$, a multitude of steady and dynamic patterns can form from random initial conditions.

**How to evaluate a new simulator operating in this space:** It would be impressive if a simulator—trained only on some of the patterns produced by a subset of the $(f, k)$ parameter space—could perform well on an unseen set of parameter values $(f, k)$ that produce fundamentally different patterns. Stability for steady-state patterns over long rollout times would also be impressive.
>> PlanetSWE

**One line description of the data:** Forced hyperviscous rotating shallow water on a sphere with earth-like topography and daily/annual periodic forcings. 

**Longer description of the data:** The shallow water equations are fundamentally a 2D approximation of a 3D flow in the case where horizontal length scales are significantly longer than vertical length scales. They are derived from depth-integrating the incompressible Navier-Stokes equations. The integrated dimension then only remains in the equation as a variable describing the height of the pressure surface above the flow. These equations have long been used as a simpler approximation of the primitive equations in atmospheric modeling of a single pressure level, most famously in the Williamson test problems. This scenario can be seen as similar to Williamson Problem 7 as we derive initial conditions from the hPa 500 pressure level in ERA5. These are then simulated with realistic topography and two levels of periodicity. 

**Associated paper**: [Paper](https://openreview.net/forum?id=RFfUUtKYOG)

**Domain expert**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: [Dedalus](https://dedalus-project.readthedocs.io/en/latest/), adapted from [example](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_sphere_shallow_water.html)

**Equation**: 

```math
\begin{align}
\frac{ \partial \vec{u}}{\partial t} &= - \vec{u} \cdot \nabla u - g \nabla h - \nu \nabla^4 \vec{u} - 2\Omega \times \vec{u} \\
\frac{ \partial h }{\partial t} &= -H \nabla \cdot \vec{u} - \nabla \cdot (h\vec{u}) - \nu \nabla^4h + F  
\end{align}
```
with $h$ the deviation of pressure surface height from the mean, $H$ the mean height, $\vec{u}$ the 2D velocity, $\Omega$ the Coriolis parameter, and F the forcing which is defined:

```python
def find_center(t):
    time_of_day = t / day
    time_of_year = t / year
    max_declination = .4 # Truncated from estimate of earth's solar decline
    lon_center = time_of_day*2*np.pi # Rescale sin to 0-1 then scale to np.pi
    lat_center = np.sin(time_of_year*2*np.pi)*max_declination
    lon_anti = np.pi + lon_center  #2*np.((np.sin(-time_of_day*2*np.pi)+1) / 2)*pi 
    return lon_center, lat_center, lon_anti, lat_center

def season_day_forcing(phi, theta, t, h_f0):
    phi_c, theta_c, phi_a, theta_a = find_center(t)
    sigma = np.pi/2
    coefficients = np.cos(phi - phi_c) * np.exp(-(theta-theta_c)**2 / sigma**2)
    forcing = h_f0 * coefficients
    return forcing
```

Visualization: 

![Gif](gif/planetswe.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| planetswe  | 0.0800| 0.0768 |0.930| $\mathbf{0.0624}$|


> About the data

**Dimension of discretized data:** 3024 timesteps of 256x512 images with "day" defined as 24 steps and "year" defined as 1008 in model time. 

**Fields available in the data:** height (scalar field), velocity (vector field).

**Number of trajectories:** 40 trajectories of 3 model years.

**Estimated size of the ensemble of all simulations:** 185.8 GB.

**Grid type:** Equiangular grid, polar coordinates.

**Initial conditions:** Sampled from hPa 500 level of [ERA5](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803), filtered for stable initialization and burned-in for half a simulation year. 

**Boundary conditions:** Spherical.

**Simulation time-step ($\Delta t$):** CFL-based step size with safety factor of .4. 

**Data are stored separated by ($\delta t$):** 1 hour in simulation time units.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 3024$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $\phi \in [0, 2 \pi]$, $\theta \in [0, \pi]$. 

**Set of coefficients or non-dimensional parameters evaluated:** $\nu$ normalized to mode 224. 

**Approximate time to generate the data:** 45 minutes using 64 icelake cores for one simulation. 

**Hardware used to generate the data:** 64 Icelake CPU cores.

> What is interesting and challenging about the data:

Spherical geometry and planet-like topography and forcing make for a proxy for real-world atmospheric dynamics where true dynamics are known. The dataset has annual and daily periodicity forcing models to either process a sufficient context length to learn these patterns or to be explicitly time aware. Furthermore, the system becomes stable making this a good system for exploring long run stability of models.

>> Post neutron star merger

**One line description of the data:** Simulations of the aftermath of a neutron star merger.

**Longer description of the data:** The simulations presented here are axisymmetrized snapshots of full three-dimensional general relativistic neutrino radiation magnetohydrodynamics. The plasma physics is treated with finite volumes with constrained transport for the magnetic field on a curvilinear grid. The system is closed by a tabulated nuclear equation of state assuming nuclear statistical equilibrium (NSE). The radiation field is treated via Monte Carlo transport, which is a particle method. The particles are not included in this dataset, however their effects are visible as source terms on the fluid.

**Associated paper**: The simulations included here are from a series of papers: [Paper 1](https://iopscience.iop.org/article/10.3847/1538-4365/ab09fc/pdf), [Paper 2](https://link.aps.org/accepted/10.1103/PhysRevD.100.023008), [Paper 3](https://arxiv.org/abs/1912.03378), [Paper 4](https://arxiv.org/abs/2212.10691), [Paper 5](https://arxiv.org/abs/2311.05796).

**Domain expert**: [Jonah Miller](https://www.thephysicsmill.com/), Los Alamos National Laboratory.

**Code or software used to generate the data**: Open source software [nublight](https://github.com/lanl/nubhlight)

**Equation**: See equations 1-5 and 16 of Miller, Ryan, Dolence (2019).

The fluid sector consists of the following system of equations.

```math
\begin{eqnarray}
  \partial_t \left(\sqrt{g}\rho_0 u^t\right) + \partial_i\left(\sqrt{g}\rho_0u^i\right)
  &=& 0\\
  \partial_t\left[\sqrt{g} \left(T^t_{\ \nu} + \rho_0u^t \delta^t_\nu\right)\right]
  + \partial_i\left[\sqrt{g}\left(T^i_{\ \nu} + \rho_0 u^i \delta^t_\nu\right)\right]
  &=& \sqrt{g} \left(T^\kappa_{\ \lambda} \Gamma^\lambda_{\nu\kappa} + G_\nu\right)\ \forall \nu = 0,1,\ldots,4\\
  \partial_t \left(\sqrt{g} B^i\right) + \partial_j \left[\sqrt{g}\left(b^ju^i - b^i u^j\right)\right] &=& 0\\
  \partial_t\left(\sqrt{g}\rho_0 Y_e u^t\right) + \partial_i\left(\sqrt{g}\rho_0Y_eu^i\right)
  &=& \sqrt{g} G_{\text{ye}}\\
\end{eqnarray}
```

The standard radiative transfer equation is
```math
\begin{equation}
    \frac{D}{d\lambda}\left(\frac{h^3\mathcal{I}_{\nu,f}}{\varepsilon^3}\right) = \left(\frac{h^2\eta_{\nu,f}}{\varepsilon^2}\right) - \left(\frac{\varepsilon \chi_{\nu,f}}{h}\right) \left(\frac{h^3\mathcal{I}_{\nu,f}}{\varepsilon^3}\right),
\end{equation}
```

<div style="transform: rotate(90deg);">
  <img src="https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/post_neutron_star_merger/gif/Ye_normalized.gif" alt="Rotated GIF">
</div>


| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| post_neutron_star_merger  | 1380 | 337 | - |-|

Preliminary benchmarking, in VRMSE. Unet and CNextU-net results are not available as these architectures needs all dimensions of the data to be multiples of 2.


> About the data 

**Dimension of discretized data:** 181 time-steps of $192 \times 128 \times 66$ snapshots.

**Fields available in the data:** fluid density (scalar field), fluid internal energy (scalar field), electron fraction (scalar field), temperate (scalar field), entropy (scalar field), velocity (vector field), magnetic field (vector field), contravariant tensor metric of space-time (tensor field, no time-dependency).

A description of fields available in an output file can be found here:
https://github.com/lanl/nubhlight/wiki

**Number of trajectories:** Currently eight full simulations. 

**Size of the ensemble of all simulations:** 110.1 GB.

**Grid type**: Uniform grid, log-spherical coordinates.

**Initial conditions:** Constant entropy torus in hydrostatic equilibrium orbiting a black hole. Black hole mass and spin, as well as torus mass, spin, electron fraction, and entropy vary.

**Boundary conditions:** open.

**Simulation time-step:** approximately 0.01 in code units. Physical time varies; roughly 147 nanoseconds for fiducial model.

**Data are stored separated by ($\Delta t$):** 50 in code units. Physical time varies; roughly 0.6 milliseconds for fiducial model.

**Total time range ($t_{min}$ to $t_{max}$):** 10000 in code units. Physical time varies; roughly 127 milliseocnds for fudicial model

**Spatial domain size ($L_x$, $L_y$, $L_z$):** Spherical coordinates. Radius roughly 2 to 1000 in code units. Physical values vary. Outer boundary is at roughly 4000 for fiducial model. Polar angle 0 to pi. Azimuthal angle 0 to 2*pi. Note that the coordinates are curvilinear. In Cartesian space, spacing is logarithmic in radius and there is a focusing of grid lines near the equator.

**Set of coefficients or non-dimensional parameters evaluated:** Black hole spin parameter a, ranges 0 to 1. Initial mass and angular momentum of torus. In dimensionless units, evaluated as inner radius Rin and radius of maximum pressure Rmax. Torus initial electron fraction Ye and entropy kb. Black hole mass in solar masses.

**Approximate time to generate the data:** Roughly 3 weeks per simulation on 300 cores.

**Hardware used to generate the data and precision used for generating the data:** Data generated at double precision on several different supercomputers. All calculations were CPU calculations parallelized with a hybrid MPI + OpenMP strategy. 1 MPI rank per socket. Oldest calculations performed on the Los Alamos Badger cluster, now decommissioned. Intel Xeon E5-2695v5 2.1 GHz. 12 cores per socket, 24 core cores per node. Simulations run on 33 nodes. Some newer simulations run on Los Alamos Capulin cluster, now decomissioned. ARM ThunderX2 nodes. 56 cores per node. Simulation run on 33 nodes.

## Simulation Index

| Scenario | Shorthand name | Description                                                         |
|----------|----------------|---------------------------------------------------------------------|
| 0        | collapsar_hi   | Disk resulting from collapse of massive rapidly rotating star.       |
| 1        | torus_b10      | Disk inspired by 2017 observation of a neutron star merger. Highest magnetic field strength. |
| 2        | torus_b30      | Disk inspired by 2017 observation of a neutron star merger. Intermediate magnetic field strength. |
| 3        | torus_gw170817 | Disk inspired by 2017 observation of a neutron star merger. Weakest magnetic field strength. |
| 4        | torus_MBH_10   | Disk from black hole-neutron star merger. 10 solar mass black hole.  |
| 5        | torus_MBH_2p31 | Disk from black hole-neutron star merger. 2.31 solar mass black hole.|
| 6        | torus_MBH_2p67 | Disk from black hole-neutron star merger. 2.76 solar mass black hole.|
| 7        | torus_MBH_2p69 | Disk from black hole-neutron star merger. 2.79 solar mass black hole.|
| 8        | torus_MBH_6    | Disk from black hole-neutron star merger. 6 solar mass black hole.   |


## General relativistic quantities
The core quantity that describes the curvature of spacetime and its
impact on a simulation is `['t0_fields']['gcon']` of the HDF5 file. From this other
quantities can be computed.

## To reproduce
The values in `simulation_parameters.json` are sufficient to reproduce a
simulation using [nubhlight](https://github.com/lanl/nubhlight) using
the `torus_cbc` problem generator, with one exception. You must
provide tabulated equation of state and opacity data. We use the SFHo
equation of state provided on the
[stellar collapse website](https://stellarcollapse.org/).
Tabulated neutrino opacities were originally computed for the Fornax
code and are not public. However adequate open source substitutes may
be generated by the [nulib](http://www.nulib.org/) library.

## Explanation of simulation parameters

Here we include, for completeness, a description of the different simulation parameters. which cover the simulation parameters chosen. Their value for each simulation is stored in `simulation_parameters.json`.

- `B_unit`, the unit of magnetic field strength. Multiplying code quantity by `B_unit` converts the quantity to units of Gauss.
- `DTd`, dump time cadence.
- `DTl`, log output time cadence.
- `DTp`, permanent restart file time cadence.
- `DTr`, temporary restart file time cadence.
- `Ledd`, (Photon) Eddington luminosity based on black hole mass.
- `L_unit`, length unit. Multiplying code quantity by `L_unit` converts it into units of cm.
- `M_unit`, mass unit. Multiplying code quantity by `M_unit` converts it into units of g.
- `Mbh`, black hole mass in units of g.
- `MdotEdd`, (Photon) Eddington accretion rate based on black hole mass.
- `N1`, number of grid points in X1 (radial) direction.
- `N2`, number of grid points in X2 (polar) direction.
- `N3`, number of grid points in X3 (azimuthal) direction.
- `PATH`, output directory for the original simulation.
- `RHO_unit`, density unit. Multiplying code quantity by `RHO_unit` converts it into units of g/cm^3.
- `Reh`, radius of the event horizon in code units.
- `Rin`, radius of the inner boundary in code units.
- `Risco`, radius of the innermost stable circular orbit in code units.
- `Rout_rad`, outer radius of neutrino transport.
- `Rout_vis`, radius used for 3D volume rendering.
- `TEMP_unit`, temperature unit. Converts from MeV (code units) to Kelvin.
- `T_unit`, time unit. Converts from code units to seconds.
- `U_unit`, energy density unit. Multiplying code quantity by `U_unit` converts it into units of erg/cm^3.
- `a`, dimensionless black hole spin. 
- `cour`, dimensionless CFL factor used to set the timestep based on the grid spacing.
- `dx`, array of grid spacing in code coordinates. (Uniform.)
- `maxnscatt`, maximum number of scattering events per superphoton particle
- `mbh`, black hole mass in solar masses.
- `hslope`, `mks_smooth`, `poly_alpha`, `poly_xt` focusing terms used for coordinate transforms
- `startx`, array of starting coordinate values for `X1`,`X2`,`X3` in code coordinates.
- `stopx`, array of ending coordinate values for `X1`,`X2`,`X3` in code coordinates.
- `tf`, final simulation time.
- `variables` list of names of primitive state vector.

> What is interesting and challenging about the data:
**What phenomena of physical interest are catpured in the data:** The 2017 detection of the in-spiral and merger of two neutron stars
was a landmark discovery in astrophysics. Through a wealth of
multi-messenger data, we now know that the merger of these
ultracompact stellar remnants is a central engine of short gamma ray
bursts and a site of r-process nucleosynthesis, where the heaviest
elements in our universe are formed. The radioactive decay of unstable
heavy elements produced in such mergers powers an optical and
infra-red transient: The kilonova.

One key driver of nucleosynthesis and resultant electromagnetic
afterglow is wind driven by an accretion disk formed around the
compact remnant. Neutrino transport plays a key role in setting the
electron fraction in this outflow, thus controlling the
nucleosynthesis.

Collapsars are black hole accretion disks formed after the core of a
massive, rapidly rotating star collapses to a black hole. These
dramatic systems rely on much the same physics and modeling as
post-merger disks, and can also be a key driver of r-processes
nucleosynthesis.

**How to evaluate a new simulator operating in this space:** The electron fraction of material blown off from the disk is the core
"delivarable." It determines how heavy elements are synthesized, which
in turn determines the electromagnetic counterpart as observed on
Earth. This is the most important piece to get right from an emulator.


>> Rayleigh Bénard convection

**One line description of the data:** 2D horizontally-periodic Rayleigh-Benard convection.

**Longer description of the data:** 
Rayleigh-Bénard convection involves fluid dynamics and thermodynamics, seen in a horizontal fluid layer heated from below, forming convective cells due to a temperature gradient. With the lower plate heated and the upper cooled, thermal energy creates density variations, initiating fluid motion. This results in Bénard cells, showcasing warm fluid rising and cool fluid descending. The interplay of buoyancy, conduction, and viscosity leads to complex fluid motion, including vortices and boundary layers. 


**Associated paper**: [Paper 1](https://www.tandfonline.com/doi/pdf/10.1080/14786441608635602), [Paper 2](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/steady-rayleighbenard-convection-between-noslip-boundaries/B4F358EB0AE83BBE9D85968DC5DDD64D
).
 
**Data generated by**: [Rudy Morel](https://www.di.ens.fr/rudy.morel/), CCM, Flatiron Institute.

**Code or software used to generate the data**: [Github repository](https://github.com/RudyMorel/the_well_rbc_sf), [Dedalus]( https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html)

**Equation**:

While we solve equations in the frequency domain, the original time-domain problem is 
```math
\begin{align*}
\frac{\partial b}{\partial t} - \kappa\,\Delta b & = -u\nabla b\,,
\\
\frac{\partial u}{\partial t} - \nu\,\Delta u + \nabla p - b \vec{e}_z & = -u \nabla u\,, 
\end{align*}
```
where $\Delta = \nabla \cdot \nabla$ is the spatial Laplacian, $b$ is the buoyancy, $u = (u_x,u_y)$ the (horizontal and vertical) velocity, and $p$ is the pressure, $\vec{e}_z$ is the unit vector in the vertical direction, with the additional constraints $\int p = 0$ (pressure gauge).

The boundary conditions vertically are as follows:
```math
\begin{align*}
b(z=0) = Lz ~~~,~~~ b(z=Lz) = 0
\\
u(z=0) = u(z=Lz) = 0
\end{align*}
```

These PDE are parameterized by the Rayleigh and Prandtl numbers through $\kappa$ and $\nu$.
```math
\begin{align*}
\text{(thermal diffusivity)} ~~~~~~~ \kappa & = \big(\text{Rayleigh} * \text{Prandtl}\big)^{-\frac12}
\\
\text{(viscosity)} ~~~~~~~ \nu & = \bigg(\frac{\text{Rayleigh}}{\text{Prandtl}}\bigg)^{-\frac12}.
\end{align*}
```

<div style="transform: rotate(90deg);">
  <img src="https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/rayleigh_benard/gif/buoyancy_normalized.gif" alt="Rotated GIF">
</div>


| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| rayleigh_benard | 0.290  | 0.361 |0.564|$\mathbf{0.149}$|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** $200$ timesteps of 
$512\times128$ images.

**Fields are available in the data:** buoyancy (scalar vield), pressure (scalar field), velocity (vector field).

**Number of simulations:** $1750$ ($35$ PDE parameters $\times$ $50$ initial conditions).

**Size of the ensemble of all simulations:** 358.4 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** the buoyancy is composed of a dumped noise added to a linear background  $b(t=0) = (Lz-z)\times\delta b_0 + z(Lz-z) \times\epsilon$ where $\epsilon$ is a Gaussian white noise of scale $10^{-3}$.
The other fields $u$ and $p$ are initialized to $0$.

**Boundary conditions:** periodic on the horizontal direction, Dirichlet conditions on the vertical direction.

**Simulation time-step:** 0.25.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 50$.

**Spatial domain size:** $0 \leq x \leq 4$ horizontally, and $0 \leq z \leq 1$ vertically.

**Set of coefficients or non-dimensional parameters evaluated:** $\text{Rayleigh}\in[1e6,1e7,1e8,1e9,1e10], \text{Prandtl}\in[0.1,0.2,0.5,1.0,2.0,5.0,10.0]$. For initial conditions $\delta b_0\in[0.2,0.4,0.6,0.8,1.0]$, the seed used to generate the initial Gaussian white noise are $40,\ldots,49$.

**Approximate time to generate the data: per input parameter** from $\sim6\,000s$ to $\sim 50\,000s$ (high Rayleigh numbers take longer), total: $\sim 60$ hours.

**Hardware used to generate the data and precision used for generating the data:** 12 nodes of 64 CPU cores with 8 processes per node, in single precision.

> What is interesting and challenging about the data:

Rayleigh-Bénard convection datasets offer valuable insights into fluid dynamics under thermal gradients, revealing phenomena like thermal plumes and turbulent eddies. Understanding these dynamics is crucial for applications in engineering and environmental science.



>> Rayleigh-Taylor instability

**One line description of the data:** Effect of spectral shape and component phases on development of Rayleigh-Taylor turbulence.

**Longer description of the data:** We consider the Rayleigh-Taylor instability for a range of Atwood numbers and initial perturbations, all of which have a log—normal horizontal energy spectrum with random phase. The dataset examines how varying the mean, standard deviation and the disparity of the random phase effects the transition to and statistics of the ensuing turbulent flow. 

**Associated paper**: [Paper](https://www.researchgate.net/publication/243660629_Reynolds_number_effects_on_Rayleigh-Taylor_instability_with_possible_implications_for_type_Ia_supernovae).

**Domain experts**: [Stefan Nixon](https://www.maths.cam.ac.uk/person/ssn34), University of Cambridge, [Romain Watteaux](https://www.linkedin.com/in/romain-watteaux-978b08162/?locale=en_US), CEA DAM, [Suart B. Dalziel](https://scholar.google.com/citations?user=OJcK5CAAAAAJ&hl=en), University of Cambridge.

**Code or software used to generate the data**: [TurMix3D](https://theses.hal.science/tel-00669707/document)

**Equation**:The flow is governed by equations for continuity, momentum and incompressibility in the case of miscible fluids with common molecular diffusivity:
```math
\begin{align}
    \partial_t\rho + \nabla\cdot(\rho \vec{u}) = 0,\\
    \partial_t(\rho \vec{u})+\nabla\cdot(\rho \vec{u} \vec{u}) = -\nabla p + \nabla\cdot\vec{\tau}+\rho \vec{g},\\
     \nabla\cdot\vec{u} = -\kappa\nabla\cdot\left(\frac{\nabla\rho}{\rho}\right). 
\end{align}
```

Here, $\rho$ is density, $\vec{u}$ is velocity, $p$ is pressure, $\vec{g}$ is gravity, $\kappa$ is the coefficient of molecular diffusivity and $\vec{\tau}$ is the deviatoric stress tensor 
```math
\begin{equation}
    \vec{\tau}= \rho\nu\left(\nabla\vec{u}+\left(\nabla\vec{u}\right)^T-\frac{2}{3}\left(\nabla\cdot\vec{u} \right)\vec{I}\right), 
\end{equation}
```
where $\nu$ is the kinematic viscosity and $\vec{I}$ is the identity matrix. 

<div style="transform: rotate(270deg);">
  <img src="https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/rayleigh_taylor_instability/gif/density_normalized.gif" alt="Rotated GIF">
</div>

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| rayleigh_taylor_instability (At = 0.25) | $\mathbf{0.875}$  | 1.57 |4.61|0.991|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** 60 time-steps of $128\times 128\times 128$ cubes.

**Fields available in the data:** Density (scalar field), velocity (vector field).

**Number of trajectories:** 45 trajectories.

**Estimated size of the ensemble of all simulations:** 255.6 GB.

**Grid type:** uniform grid, cartesian coordinates.

**Initial conditions:** Initial conditions have been set by imposing a log—normal profile for the shape of energy spectrum in wavenumber space, such that:
$$A(k) = \frac{1}{k\sigma\sqrt{2\pi}} \exp\Big(-\frac{(\ln (k) - \mu)^2}{2\sigma^2}\Big) \quad\textrm{with}\quad k = \sqrt{k^2_x+k^2_y}$$
where $\mu$ is the mean and $\sigma$ is the standard deviation of the profile. Furthermore, we have imposed a random phase to the corresponding complex Fourier component (i.e. a random value for the argument of the complex Fourier component) between zero and a varied maximum ($\phi_{max}$), finally after Fourier transforming to physical space the mean of the resulting profile is normalized to $3.10^5$ to ensure comparable power. 


**Boundary conditions:**** Periodic boundary conditions on sides walls and slip conditions on the top and bottom walls.

**Simulation time-step:** $\Delta t$ is set such that the maximum Courant number is $\frac12(CFL_{max}=0.5)$. Therefore, the time step decreases as the flow accelerates.

**Data are stored separated by ($\Delta t$):**  different according to At number.

The time difference between frames varies as the flow accelerates, thus the largest occur at the beginning of the simulation ($\delta t \sim 5s$) and the smallest at the end ($\delta t \sim 0.1s$).

**Total time range ($t_{min}$ to $t_{max}$):** Varies from $t_{min}=0$ to $t_{max}$ between $\sim 30s$ and $\sim 100s$, depending on Atwood number.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $[0,1]\times[0,1]\times[0,1]$.

**Set of coefficients or non-dimensional parameters evaluated:** We run simulations with 13 different initializations for five different Atwood number $At\in {\frac34, \frac12, \frac14, \frac18, \frac{1}{16}}$. The first set on initial conditions considers varying the mean $\mu$ and standard deviation $\sigma$ of the profile $A(k)$ with $\mu\in{1, 4, 16}$ and $\sigma\in{\frac14, \frac12, 1}$, the phase (argument of the complex Fourier component) $\phi$ was set randomly in the range $[0,2\pi)$. The second set of initial conditions considers a fixed mean ($\mu=16$) and standard deviation ($\sigma =0.25$) and a varieed range of random phases (complex arguments $\phi\in[0,\phi_{max}$)) given to each Fourier component. The four cases considered are specified by $\phi_{max}\in { \frac{\pi}{128}, \frac{\pi}{8}, \frac{\pi}{2}, \pi}$. 

**Approximate time to generate the data:** 1 hour on 128 CPU cores for 1 simulation. 65 hours on 128 CPU cores for all simulations.

**Hardware used to generate the data:** 128 CPU core on the Ocre supercomputer at CEA, Bruyères-le-Châtel, France.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** In this data there are three key aspects of physical interest. Firstly, impact of coherence on otherwise random initial conditions. Secondly, the effect of the shape of the initial energy spectrum on the structure of the flow. Finally, the transition from the Boussinesq to the non-Boussinesq regime where the mixing width transitions from symmetric to asymmetric growth.  

**How to evaluate a new simulator operating in this space:**

From a fundamental standpoint we, would expect the density field to be advected and mixed rather than created or destroyed to give appropriate statistics. From a qualitative perspective, given that the underlying simulations are of comparable spatial resolution to the simulations run by the alpha group (Dimonte et. al. 2003) we would consider a good emulator to produce a comparable value for α as reported in their paper for an appropriately similar set of initial conditions. This parameter is derived by considering the flow after the initial transient. At this stage, the width of the turbulent mixing zone, $L$, is self-similar and grows as $L= \alpha \* At \* g \* t^2$. They reported a value of $\alpha$=0.025±0.003. In addition, during this self-regime, we would expect to observe energy spectra with a similar shape to those reported in Cabot and Cook 2006, specifically exhibiting an appropriate $k^{-\frac53}$ cascade. From a structural perspective, we would expect that for an initialization with a large variety of modes in the initial spectrum to observe a range of bubbles and spikes (upward and downward moving structures), whereas in the other limit (where this only on mode in the initial spectrum) we expect to observe a single bubble and spike.  In addition, a good emulator would exhibit symmetric mixing with for low Atwood numbers in the Boussinesq regime (defined as $At$ < 0.1 by Andrews and Dalziel 2010) and asymmetries in the mixing with for large Atwood number. 



>> Periodic shear flow

**One line description of the data:** 2D periodic incompressible shear flow. 

**Longer description of the data:** 
Shear flow are a type of fluid characterized by the continuous deformation of adjacent fluid layers sliding past each other with different velocities. This phenomenon is commonly observed in various natural and engineered systems, such as rivers, atmospheric boundary layers, and industrial processes involving fluid transport.
The dataset explores a 2D periodic shearflow governed by incompressible Navier-Stokes equation. 

**Associated paper**: [Paper 1](https://www.sciencedirect.com/book/9780124059351/fluid-mechanics), [Paper 2](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.100504), [Paper 3](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.9.014202)

**Data generated by**: [Rudy Morel](https://www.di.ens.fr/rudy.morel/), CCM, Flatiron Institute.

**Code or software used to generate the data**: [Github repository](https://github.com/RudyMorel/the_well_rbc_sf), [Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_shear_flow.html)

**Equation**:

While we solve equations in the frequency domain, the original time-domain problem is 
```math
\begin{align*}
\frac{\partial u}{\partial t} + \nabla p - \nu \Delta u & = -u\cdot\nabla u\,,
\\
\frac{\partial s}{\partial t} - D \Delta s & = -u \cdot\nabla s\,, 
\end{align*}
```
where $\Delta = \nabla \cdot \nabla$ is the spatial Laplacian, $u = (u_x,u_y)$ is the (horizontal and vertical) velocity, $s$ is the tracer, and $p$ is the pressure, 
with the additional constraints $\int p = 0$ (pressure gauge).

These PDE are parameterized by the Reynolds and Schmidt numbers through $\nu$ and $D$.
```math
\begin{align*}
\text{(viscosity)} ~~~~~~~ \nu & = 1 \, / \, \text{Reynolds}
\\
\text{(diffusivity)} ~~~~~~~ D & = \nu \, / \, \text{Schmidt}
\end{align*}
```
The tracer is passive and here for visualization purposes only.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/shear_flow/gif/tracer_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| shear_flow  | 0.102 | 0.0954 |0.161|$\mathbf{0.0530}$|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** $200$ time-steps of $128\times256$ images.

**Fields available in the data:** tracer (scalar field), velocity (vector field), pressure (scalar field).

**Number of simulations:** $1120$ ($28$ PDE parameters $\times$ $40$ initial conditions).

**Size of the ensemble of all simulations:** 114.7 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** the shear field $u_1$ is composed of $n_\text{shear}$ shears uniformly spaced along the $z$ direction. Each shear is implemented with a tanh (hyperbolic tangent) $\text{tanh}(5\frac{y-y_k}{n_\text{shear}w})$ where $z_k$ is the vertical position of the shear and $w$ is a width factor.
The velocity field $u_2$ is composed of sinusoids along the $x$ direction located at the shear. These sinusoids have an exponential decay away from the shear in the $y$ direction $\text{sin}(n_\text{blobs}\pi x)\,e^{\frac{25}{w^2}|y-y_k|^2}$.
The tracer matches the shear at initialization. The pressure is initialized to zero.
The initial condition is thus indexed by $n_\text{shear},n_\text{blobs},w$.

**Boundary conditions:** periodic.

**Simulation time-step:** 0.1.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 20$.

**Spatial domain size:** $0\leq x \leq 1$ horizontally, and $-1 \leq y \leq 1$ vertically.

**Set of coefficients or non-dimensional parameters evaluated:** $\text{Reynolds}\in[1e4, 5e4, 1e5, 5e5], \text{Schmidt}\in[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]$. For initial conditions $n_\text{shear}\in[2,4]$, $n_\text{blobs}\in[2,3,4,5]$, $w\in[0.25, 0.5, 1.0, 2.0, 4.0]$.


**Approximate time to generate the data:** per input parameter: $\sim 1500s$, total: $\sim 5$ hours.

**Hardware used to generate the data and precision used for generating the data:** 7 nodes of 64 CPU cores each with 32 tasks running in parallel on each node, in single precision.

> What is interesting and challenging about the data:

Shear flow are non-linear phenomena arrising in fluid mechanics and turbulence.
Predicting the behavior of the shear flow under different Reynolds and Schmidt numbers is essential for a number of applications in aerodynamics, automotive, biomedical. 
Furthermore, such flow are unstable at large Reynolds number. 



>> Supernova Explosion in Turbulent Interstellar medium in galaxies

**One line description of the data:** 
Blastwave in dense cool gas cloud.

**Longer description of the data:** 
The simulations solve an explosion inside a compression of a monatomic ideal gas, which follows the equation of state with the specific heat ratio $\gamma=5/3$.
The gas in these simulations mocks interstellar medium in the Milky Way Galaxy.
At the beginning of the simulations, the thermal energy of a supernova is dumped at the center of the simulation box.
The hot ($\sim 10^7$ K) gas is immediately accelerated and makes the blastwave.
Because velocities of the hot gas become supersonic, much fine resolution and small timestep are required to resolve the dynamics.
The physical quantities are also distributed in seven orders of magnitude, which requires a large number of simulation steps.

**Associated paper**: [Paper 1](https://academic.oup.com/mnras/article/526/3/4054/7316686), [Paper 2](https://arxiv.org/abs/2311.08460).

**Domain expert**:[Keiya Hirashima](https://kyafuk.github.io/utokyo-hirashima/index.html), University of Tokyo & CCA, Flatiron Institute.

**Code or software used to generate the data**: ASURA-FDPS (Smoothed Particle Hydrodynamics), [Github repository](https://github.com/FDPS/FDPS)

**Equation**:

```math
\begin{align}
P&=(\gamma-1) \rho u \\
\frac{d \rho}{dt} &= -\rho \nabla \cdot \boldsymbol{v} \\
\frac{d^2 \boldsymbol{r}}{dt^2}  &= -\frac{\nabla P}{\rho} + \boldsymbol{a}_{\rm visc}-\nabla \Phi \\
\frac{d u}{dt} &= -\frac{P}{\rho} \nabla \cdot \boldsymbol{v} + \frac{\Gamma-\Lambda}{\rho}
\end{align}
```

where $P$, $\rho$, and $u$ are the pressure. $r$ is the position, $a_{\rm visc}$ is the acceleration generated by the viscosity, $\Phi$ is the gravitational potential, $\Gamma$ is the radiative heat influx per unit volume, and $\Lambda$ is the radiative heat outflux per unit volume.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/supernova_explosion_128/gif/temperature_normalized.gif)


> About the data

**Dimension of discretized data** $59$ time-steps of  $128\times 128\times 128$ cubes.

**Fields available in the data:**
Pressure (scalar field), density (scalar field), temperature(scalar field), velocity (tensor field).

**Number of trajectories:** 260.

**Estimated size of the ensemble of all simulations:** 754 GB


**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** $820$ random seeds generated using https://github.com/amusecode/amuse/blob/main/src/amuse/ext/molecular_cloud.py (Virialized isothermal gas sphere with turbulence following the velocity spectrum $E(k) \propto k^{-2}$, which is Burgers turbulence ([Burgers 1948](https://www.sciencedirect.com/science/article/abs/pii/S0065215608701005) and [Kupilas+2021](https://doi.org/10.1093/mnras/staa3889) for reference ))

**Boundary conditions:** open.

**Data are stored separated by ($\Delta t$):** $100$ ~ $10 000$ years (variable timesteps).

**Total time range ($t_{min}$ to $t_{max}$):** $0$ yr to $0.2$ Myr.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** 60 pc.

**Set of coefficients or non-dimensional parameters evaluated:** Initial temperature $T_0$=\{100K\}, Initial number density of hydrogen $\rho_0=$\{44.5/cc\}, metallicity (effectively strength of cooling) $Z=\{Z_0\}$.

**Approximate time to generate the data (CPU hours):**
|  0.1 M $\odot$ |
|:----------:|
|  $3500$ |

**Hardware used to generate the data and precision used for generating the data:** up to 1040 CPU cores per run. 



> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:**
The simulations are designed as an supernova explosion, which is the explosion at the last moment of massive stars, in a high-density starforming molecular cloud with a large density contrast. An adiabatic compression of a monatomic ideal gas is assumed.
To mimic the explosion, the huge thermal energy ($10^{51}$ erg) is injected at the center of the calculation box and going to make the blastwave, which sweeps out the ambient gas and shells called as supernova feedback. These interactions between supernovae and surrounding gas are interesting because stars are formed in dense and cold regions.

However, calclatig the propagation of blastwaves requires tiny timesteps to calculate and numerous integration steps. When supernova feedback is incorporated in a galaxy simulation, some functions fitted using local high resolution simulations have been used.

**How to evaluate a new simulator operating in this space:**
In context of galaxy simulations, the time evolution of thermal energy and momentum are important. We note that those physical quantities are not necessarily conserved because radiative cooling and heating are considered and thermal energy is seamlessly being converted into momentum.

>> Supernova Explosion in Turbulent Interstellar medium in galaxies

**One line description of the data:** 
Blastwave in dense cool gas cloud.

**Longer description of the data:** 
The simulations solve an explosion inside a compression of a monatomic ideal gas, which follows the equation of state with the specific heat ratio $\gamma=5/3$.
The gas in these simulations mocks interstellar medium in the Milky Way Galaxy.
At the beginning of the simulations, the thermal energy of a supernova is dumped at the center of the simulation box.
The hot ($\sim 10^7$ K) gas is immediately accelerated and makes the blastwave.
Because velocities of the hot gas become supersonic, much fine resolution and small timestep are required to resolve the dynamics.
The physical quantities are also distributed in seven orders of magnitude, which requires a large number of simulation steps.

**Associated paper**: [Paper 1](https://academic.oup.com/mnras/article/526/3/4054/7316686), [Paper 2](https://arxiv.org/abs/2311.08460).

**Domain expert**: [Keiya Hirashima](https://kyafuk.github.io/utokyo-hirashima/index.html), University of Tokyo & CCA, Flatiron Institute.

**Code or software used to generate the data**: ASURA-FDPS (Smoothed Particle Hydrodynamics), [Github repository](https://github.com/FDPS/FDPS)

**Equation**:

```math
\begin{align}
P&=(\gamma-1) \rho u \\
\frac{d \rho}{dt} &= -\rho \nabla \cdot \mathbf{v} \\
\frac{d^2 \mathbf{r}}{dt^2}  &= -\frac{\nabla P}{\rho} + \mathbf{a}_{\rm visc}-\nabla \Phi \\
\frac{d u}{dt} &= -\frac{P}{\rho} \nabla \cdot \mathbf{v} + \frac{\Gamma-\Lambda}{\rho}
\end{align}
```

where $P$, $\rho$, and $u$ are the pressure. $r$ is the position, $a_{\rm visc}$ is the acceleration generated by the viscosity, $\Phi$ is the gravitational potential, $\Gamma$ is the radiative heat influx per unit volume, and $\Lambda$ is the radiative heat outflux per unit volume.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/supernova_explosion_64/gif/temperature_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| supernova_explosion_64  | 1.01 | 1.01 |1.06|$\mathbf{0.968}$|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data** $59$ time-steps of  $64\times 64\times 64$ cubes.

**Fields available in the data:**
Pressure (scalar field), density (scalar field), temperature(scalar field), velocity (tensor field).

**Number of trajectories:** 740.

**Estimated size of the ensemble of all simulations:** 268.2 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** $820$ random seeds generated using https://github.com/amusecode/amuse/blob/main/src/amuse/ext/molecular_cloud.py (Virialized isothermal gas sphere with turbulence following the velocity spectrum $E(k) \propto k^{-2}$, which is Burgers turbulence ([Burgers 1948](https://www.sciencedirect.com/science/article/abs/pii/S0065215608701005) and [Kupilas+2021](https://doi.org/10.1093/mnras/staa3889) for reference).

**Boundary conditions:** open.

**Data are stored separated by ($\Delta t$):** $100$ ~ $10,000$ years (variable timesteps). [CHECK]

**Total time range ($t_{min}$ to $t_{max}$):** $0$ yr to $0.2$ Myr.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** 60 pc.

**Set of coefficients or non-dimensional parameters evaluated:** Initial temperature $T_0$=\{100K\}, Initial number density of hydrogen $\rho_0=$\{44.5/cc\}, metallicity (effectively strength of cooling) $Z=\{Z_0\}$.

**Approximate time to generate the data (CPU hours):**
| 1M $_\odot$ | 0.1 M $\odot$ |
|:----------:|:----------:|
| $300$ | $3500$ |

**Hardware used to generate the data and precision used for generating the data:** up to 1040 CPU cores per run.




> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:**
The simulations are designed as an supernova explosion, which is the explosion at the last moment of massive stars, in a high-density starforming molecular cloud with a large density contrast. An adiabatic compression of a monatomic ideal gas is assumed.
To mimic the explosion, the huge thermal energy ($10^{51}$ erg) is injected at the center of the calculation box and going to make the blastwave, which sweeps out the ambient gas and shells called as supernova feedback. These interactions between supernovae and surrounding gas are interesting because stars are formed in dense and cold regions.

However, calclating the propagation of blastwaves requires tiny timesteps to calculate and numerous integration steps. When supernova feedback is incorporated in a galaxy simulation, some functions fitted using local high resolution simulations have been used.

**How to evaluate a new simulator operating in this space:**
In context of galaxy simulations, the time evolution of thermal energy and momentum are important. We note that those physical quantities are not necessarily conserved because radiative cooling and heating are considered and thermal energy is seamlessly being converted into momentum.

>> Turbulent Interstellar medium in galaxies

**One line description of the data:**  Turbulence in interstellar medium in various evolution stages of galaxies.

**Longer description of the data:**  These simulations are a turbulent fluid with gravity modeling interstellar medium in galaxies. These fluids make dense filaments, which will form new stars. The timescale and frequency of making new filaments are varied depending on the strength of cooling. It is parametrized by the amount of metal (metallicity), density, and temperature.

**Associated paper**: [Paper](https://academic.oup.com/mnras/article/526/3/4054/7316686).

**Domain expert**: [Keiya Hirashima](https://kyafuk.github.io/utokyo-hirashima/index.html), University of Tokyo & CCA, Flatiron Institute.

**Code or software used to generate the data**: ASURA-FDPS (Smoothed Particle Hydrodynamics), [Github repository](https://github.com/FDPS/FDPS)

**Equation**: 

```math
\begin{align}
P&=(\gamma-1) \rho u \\
\frac{d \rho}{dt} &= -\rho \nabla \cdot \mathbf{v} \\
\frac{d^2 \mathbf{r}}{dt^2}  &= -\frac{\nabla P}{\rho} + \mathbf{a}_{\rm visc}-\nabla \Phi \\
\frac{d u}{dt} &= -\frac{P}{\rho} \nabla \cdot \mathbf{v} + \frac{\Gamma-\Lambda}{\rho}
\end{align}
```

where $P$, $\rho$, and $u$ are the pressure. $r$ is the position, $a_{\rm visc}$ is the acceleration generated by the viscosity, $\Phi$ is the gravitational potential, $\Gamma$ is the radiative heat influx per unit volume, and $\Lambda$ is the radiative heat outflux per unit volume.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/turbulence_gravity_cooling/gif/temperature_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| turbulence_gravity_cooling | 1.31 |2.06| 1.44 |$\mathbf{1.16}$ |

> About the data

**Dimension of discretized data:** $50$ time-steps of  $64\times 64\times 64$ cubes.

**Fields available in the data:** Pressure (scalar field), density (scalar field), temperature(scalar field), velocity (tensor field).

**Number of trajectories:** 2700 (27 parameters sets $\times$ 100 runs).

**Estimated size of the ensemble of all simulations:** 829.4 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** $2700$ random seeds generated using https://github.com/amusecode/amuse/blob/main/src/amuse/ext/molecular_cloud.py (Virialized isothermal gas sphere with turbulence following the velocity spectrum $E(k) \propto k^{-2}$, which is Burgers turbulence (Burgers 1948 and Kupilas+2021 for reference )).

**Boundary conditions:** open.

**Simulation time-step:** $2,000$ ~ $10,000$ years (variable timesteps).

**Data are stored separated by ($\Delta t$):** 0.02 free fall time.

**Total time range ($t_{min}$ to $t_{max}$):** 1 Free Fall time (= $L^3/GM$ ); $L=(\rho / \rho_0)^{1/3} \times 60$ pc, $\rho_0=44.5/\rm{cc}$, $M=1,000,000$ M $_\odot$.


**Spatial domain size ($L_x$, $L_y$, $L_z$):**

|           | Domain Length ($L$) | Free Fall Time | Snapshot ($\delta t$) |
|----------|:----------:|:----------:|:----------:|
| **Dense (44.5 cm $^{-3}$)** | 60 pc | 6.93 Myr | 0.14 Myr |
| **Moderate (4.45 cm $^{-3}$)** | 129 pc | 21.9 Myr |0.44 Myr |
| **Sparse (0.445 cm $^{-3}$)** | 278 pc | 69.3 Myr | 1.4 Myr |

**Set of coefficients or non-dimensional parameters evaluated:** Initial temperature $T_0$=\{10K, 100K, 1000K\}, Initial number density of hydrogen $\rho_0=$\{44.5/cc, 4.45/cc, 0.445/cc\}, metallicity (effectively strength of cooling) $Z=\{Z_0, 0.1Z_0, 0\}$.


**Approximate time to generate the data:** $600,000$ node hours for all simulations.

#### For dense dataset (CPU hours)
|           | Strong (1Z $_\odot$) | Weak (0.1 Z $_\odot$) | Adiabatic (0 Z $_\odot$) |
|----------:|----------:|----------:|----------:|
| **$10$ K** | $240$  | $167$ | $77$ |
| **$100$ K** | $453$ | $204$  | $84$ |
| **$1000$ K** | $933$ | $186$  | $46$ |

#### For moderate dataset (CPU hours)
|           | Strong (1Z $_\odot$) | Weak (0.1 Z $_\odot$) | Adiabatic (0 Z $_\odot$) |
|----------:|----------:|----------:|----------:|
| **$10$ K** | $214$  | $75$ | $62$ |
| **$100$ K** | $556$ | $138$  | $116$ |
| **$1000$ K** | $442$ | $208$  | $82$ |

#### For sparse dataset (CPU hours)
|           | Strong (1Z $_\odot$) | Weak (0.1 Z $_\odot$) | Adiabatic (0 Z $_\odot$) |
|----------:|----------:|----------:|----------:|
| **$10$ K** | $187$  | $102$ | $110$ |
| **$100$ K** | $620$ | $101$  | $92$ |
| **$1000$ K** | $286$ | $129$  | $93$ |



**Hardware used to generate the data and precision used for generating the data:** up to 1040 CPU cores per run.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:**
Gravity, hydrodynamics and radiative cooling/heating are considered in the simulations. Radiative cooling/heating is parameterized with metallicity, which the ratio of heavier elements than helium. The larger and metallicity corresponds to the later and early stage of galaxies and universe, respectively.
It also affects the time scale of cooling/heating and star formation rate. For instance, star formation happens at dense and cold region. With the strong cooling/heating rate, dense regions are quickly cooled down and generates new stars. Inversely, in the case of a weak cooling/heating, when gas is compressed, it is heated up and prevent new stars from being generated.

In the case of cold gas with strong cooling/heating, it can easily make dense regions, which require small timesteps and a lot of integration steps. That makes it difficult to get the resolution higher.

**How to evaluate a new simulator operating in this space:**
The new simulator should be able to detect potential regions of star formation / potential number of newborn stars, because star formation regions are very dense and need very small timesteps, which results in a massive number of calculation steps.



>> Turbulent Radiative Layer - 2D

**One line description of the data:** Everywhere in astrophysical systems hot gas moves relative to cold gas, which leads to mixing, and mixing populates intermediate temperature gas that is highly reactive—in this case it is rapidly cooling.

**Longer description of the data:** In this simulation, there is cold, dense gas on the bottom and hot dilute gas on the top. They are moving relative to each other at highly subsonic velocities. This set up is unstable to the Kelvin Helmholtz instability, which is seeded with small scale noise that is varied between the simulations. The hot gas and cold gas are both in thermal equilibrium in the sense that the heating and cooling are exactly balanced. However, once mixing occurs as a result of the turbulence induced by the Kelvin Helmholtz instability the intermediate temperatures become populated. This intermediate temperature gas is not in thermal equilibrium and cooling beats heating. This leads to a net mass flux from the hot phase to the cold phase. This process occurs in the interstellar medium, and in the Circum-Galactic medium when cold clouds move through the ambient, hot medium. By understanding how the total cooling and mass transfer scale with the cooling rate we are able to constrain how this process controls the overall phase structure, energetics and dynamics of the gas in and around galaxies.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/2041-8213/ab8d2c/pdf)

**Domain expert**: [Drummond Fielding](https://dfielding14.github.io/), CCA, Flatiron Institute & Cornell University.

**Code or software used to generate the data**: [Athena++](https://www.athena-astro.app/)

**Equation**: 

```math
\begin{align}
\frac{ \partial \rho}{\partial t} + \nabla \cdot \left( \rho \vec{v} \right) &= 0 \\
\frac{ \partial \rho \vec{v} }{\partial t} + \nabla \cdot \left( \rho \vec{v}\vec{v} + P \right) &= 0 \\
\frac{ \partial E }{\partial t} + \nabla \cdot \left( (E + P) \vec{v} \right) &= - \frac{E}{t_{\rm cool}} \\
E = P / (\gamma -1) \, \, \gamma &= 5/3
\end{align}
```
with $\rho$ the density, $\vec{v}$ the 2D velocity, $P$ the pressure, $E$ the total energy, and $t_{\rm cool}$ the cooling time.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/turbulent_radiative_layer_2D/gif/density_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| turbulent_radiative_layer_2D  | 0.967| 1.01 |0.576| 0.575|

Preliminary benchmarking, in VRMSE.


> About the data

**Dimension of discretized data:** 101 timesteps of 384x128 images.

**Fields available in the data:** Density (scalar field), pressure (scalar field), velocity (vector field).

**Number of trajectories:** 90 (10 different seeds for each of the 9 $t_{cool}$ values).

**Estimated size of the ensemble of all simulations:** 6.9 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Analytic, described in the [paper](https://ui.adsabs.harvard.edu/abs/2020ApJ...894L..24F/abstract).

**Boundary conditions:** Periodic in the x-direction, zero-gradient for the y-direction.

**Simulation time-step ($\Delta t$):** varies with $t_{cool}$. Smallest $t_{cool}$ has $\Delta t = 1.36\times10^{-2}$ and largest $t_{cool}$ has $\Delta t = 1.74\times10^{-2}$. Not that this is not in seconds. This is in dimensionless simulation time.

**Data are stored separated by ($\delta t$):** 1.597033 in simulation time.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 159.7033$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $x \in [-0.5, 0.5]$, $y \in [-1, 2]$ giving $L_x = 1$ and $L_y = 3$.

**Set of coefficients or non-dimensional parameters evaluated:** $t_{cool} = \{0.03, 0.06, 0.1, 0.18, 0.32, 0.56, 1.00, 1.78, 3.16\}$. 

**Approximate time to generate the data:** 84 seconds using 48 cores for one simulation. 100 CPU hours for everything.

**Hardware used to generate the data:** 48 CPU cores.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:**
-	The mass flux from hot to cold phase.
-	The turbulent velocities.
-	Amount of mass per temperature bin (T = press/dens).


**How to evaluate a new simulator operating in this space:** See whether it captures the right mass flux, the right turbulent velocities, and the right amount of mass per temperature bin.

>> Turbulent Radiative Mixing Layers - 3D

**One line description of the data:** In many astrophysical systems, hot gas moves relative to cold gas, which leads to mixing. Mixing populates intermediate temperature gas that is highly reactive — in this case it is rapidly cooling.

**Longer description of the data:** In this simulation, there is cold, dense gas on the bottom and hot dilute gas on the top. They are moving relative to each other at highly subsonic velocities. This set up is unstable to the Kelvin Helmholtz instability, which is seeded with small scale noise that is varied between the simulations. The hot gas and cold gas are both in thermal equilibrium in the sense that the heating and cooling are exactly balanced. However, once mixing occurs as a result of the turbulence induced by the Kelvin Helmholtz instability, the intermediate temperatures become populated. This intermediate temperature gas is not in thermal equilibrium, and cooling beats heating. This leads to a net mass flux from the hot phase to the cold phase. This process occurs in the interstellar medium, and in the Circum-Galactic medium when cold clouds move through the ambient, hot medium. By understanding how the total cooling and mass transfer scale with the cooling rate, we are able to constrain how this process controls the overall phase structure, energetics and dynamics of the gas in and around galaxies.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/2041-8213/ab8d2c/pdf)

**Domain expert**: [Drummond Fielding](https://dfielding14.github.io/), CCA, Flatiron Institute & Cornell University.

**Code or software used to generate the data**: [Athena++](https://www.athena-astro.app/)

**Equation**: 

```math
\begin{align}
\frac{ \partial \rho}{\partial t} + \nabla \cdot \left( \rho \vec{v} \right) &= 0 \\
\frac{ \partial \rho \vec{v} }{\partial t} + \nabla \cdot \left( \rho \vec{v}\vec{v} + P \right) &= 0 \\
\frac{ \partial E }{\partial t} + \nabla \cdot \left( (E + P) \vec{v} \right) &= - \frac{E}{t_{\rm cool}} \\
E = P / (\gamma -1) \, \, \gamma &= 5/3
\end{align}
```
with $\rho$ the density, $\vec{v}$ the 3D velocity, $P$ the pressure, $E$ the total energy, and $t_{\rm cool}$ the cooling time.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/turbulent_radiative_layer_3D/gif/density_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| turbulent_radiative_layer_2D  | 103 |98.0| 61.5 |73.0|

Preliminary benchmarking, in VRMSE.

> About the data

**Dimension of discretized data:** 101 timesteps of 256x128x128 arrays.

**Fields available in the data:** Density (scalar field), pressure (scalar field), velocity (vector field).

**Number of trajectories:** 90 trajectories (10 different seeds for each of the 9 $t_{cool}$ variations).

**Estimated size of the ensemble of all simulations:** 744.6 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Analytic, described in the [paper](https://ui.adsabs.harvard.edu/abs/2020ApJ...894L..24F/abstract).

**Boundary conditions:** periodic for the 128x128 directions ($x,y$), and zero-gradient for the 256 direction ($z$).

**Simulation time-step:** varies with $t_{cool}$. Smallest $t_{cool}$ is $1.32.10^{-2}$, largest $t_{cool}$ is $1.74.10^{-2}$. This is not in seconds, as this is a dimensionless simulation time. To convert, the code time is $L_{box}/cs_{hot}$, where $L_{box}$= 1 parsec and cs_{hot}=100km/s.

**Data are stored separated by ($\Delta t$):** data is separated by intervals of simulation time of 2.661722.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 266.172178$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $x,y\in\[-0.5,0.5\]$, $z\in\[-1,1\]$.

**Set of coefficients or non-dimensional parameters evaluated:** $t_{cool} = \{0.03, 0.06, 0.1, 0.18, 0.32, 0.56, 1.00, 1.78, 3.16\}$.

**Approximate time to generate the data:** $34,560$ CPU hours for all simulations.

**Hardware used to generate the data:** each simulation was generated on a 128 core "Rome" node.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** Capte the mass flux from hot to cold phase. Capture turbulent velocities. Capture the amount of mass per temperature bin ($T = \frac{P}{\rho}$).

**How to evaluate a new simulator operating in this space:** Check whether the above physical phenomena are captured by the algorithm.

>> Multistability of viscoelastic fluids in a 2D channel flow

**One line description of the data:** Multistability in viscoelastic flows, i.e. four different attractors (statistically stable states) are observed for the same set of parameters depending on the initial conditions. 

**Longer description of the data:** Elasto-inertial turbulence (EIT) is a recently discovered two-dimensional chaotic flow state observed in dilute polymer solutions. Two-dimensional direct numerical simulations show (up to) four coexistent attractors: the laminar state (LAM), a steady arrowhead regime (SAR), Elasto-inertial turbulence (EIT) and a ‘chaotic arrowhead regime’ (CAR). The SAR is stable for all parameters considered here, while the final pair of (chaotic) flow states are visually very similar and can be distinguished only by the presence of a weak polymer arrowhead structure in the CAR regime. Both chaotic regimes are maintained by an identical near-wall mechanism and the weak arrowhead does not play a role. The data set includes snapshots on the four attractors as well as two edge states. An edge state is an unstable state that exists on the boundary between two basins of attractors, the so-called edge manifold. Edge states have a single unstable direction out of the manifold and are relevant since the lie exactly on the boundary separating qualitatively different behaviours of the flow. The edge states in the present data set are obtained through edge tracking between the laminar state and EIT and between EIT and SAR. 

**Associated paper**: [Paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D63B7EDB638451A6FC2FBBFDA85E1BBD/S0022112024000508a.pdf/multistability-of-elasto-inertial-two-dimensional-channel-flow.pdf)

**Domain experts**: [Miguel Beneitez](https://beneitez.github.io/) and [Richard Kerswell](https://www.damtp.cam.ac.uk/user/rrk26/), DAMTP, University of Cambridge, UK.

**Code or software used to generate the data**: [Dedalus](https://dedalus-project.readthedocs.io/en/latest/index.html) v2.

**Equation**:

```math
\begin{align}
Re(\partial_t \mathbf{u^*} + (\mathbf{u^*}\cdot\nabla)\mathbf{u^*} ) + \nabla p^* &= \beta \Delta \mathbf{u^*} + (1-\beta)\nabla\cdot \mathbf{T}(\mathbf{C^*}),\\
\partial_t \mathbf{C^*} + (\mathbf{u^*}\cdot\nabla)\mathbf{C^*} +\mathbf{T}(\mathbf{C^*}) &= \mathbf{C^*}\cdot\nabla \mathbf{u^*} + (\nabla \mathbf{u^*})^T \cdot \mathbf{C^*} + \epsilon \Delta \mathbf{C^*}, \\
\nabla \mathbf{u^*} &= 0,\\
\textrm{with} \quad \mathbf{T}(\mathbf{C^*}) &= \frac{1}{\text{Wi}}(f(\textrm{tr}(\mathbf{C^*}))\mathbf{C^*} - \mathbf{I}),\\
\textrm{and} \quad f(s) &:= \left(1- \frac{s-3}{L^2_{max}}\right)^{-1}. 
\end{align}
```

where $\mathbf{u^\*} = (u^\*,v^\*)$ is the streamwise and wall-normal velocity components, $p^\*$ is the pressure, $\mathbf{C^\*}$ is the positive definite conformation tensor which represents the ensemble average of the produce of the end-to-end vector of the polymer molecules. In 2D, 4 components of the tensor are solved: $c\_{xx}^{\*}, c^{\*}\_{yy}, c^{\*}\_{zz}, c^{\*}\_{xy}$. $\mathbf{T}(\mathbf{C^{\*}})$ is the polymer stress tensor given by the FENE-P model.


![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/viscoelastic_instability/gif/pressure_normalized.gif)

> About the data

**Dimension of discretized data:** 
- EIT: 34 trajectories with 60 timesteps, 512x512 images (chaotic solution). 
- CAR: 39 trajectories with 60 timesteps, 512x512 images (chaotic solution).
- SAR: 20 trajectories with 20 timesteps, 512x512 images (simple periodic solutions). 
- Transition to chaos between EIT and SAR: 36 snapshots with 20 timesteps of 512x512 images. 
- Transition to non-chaotic state between EIT and SAR: 38 snapshots with 20 timesteps of 512x512 images. 
- Transition to chaos between EIT and Laminar: 43 snapshots with 20 timesteps of 512x512 images. 
- Transition to non-chaotic state between EIT and Laminar: 49 snapshots with 20 timesteps of 512x512 images.

**Fields available in the data:** pressure (scalar field), velocity (vector field), positive conformation tensor ( $c\_{xx}^{\*}, c^{\*}\_{yy},, c^{\*}\_{xy}$ are in tensor fields, $c^{\*}\_{zz}$ in scalar fields).

**Number of trajectories:** 260 trajectories.

**Estimated size of the ensemble of all simulations:** 66 GB.

**Grid type:** uniform cartesian coordinates.

**Initial conditions:**
- Edge trajectory: linear interpolation between a chaotic and a non-chaotic state. 
- SAR: continuation of the solution obtained through a linear instability at a different parameter set using time-stepping. 
- EIT: laminar state + blowing and suction at the walls. 
- CAR: SAR + blowing and suction at the walls.

**Boundary conditions:** no slip conditions for the velocity ( $(u^\*,v^\*)=(0,0)$ ) at the wall and $\epsilon=0$ at the wall for the equation for $\mathbf{C^*}$.

**Simulation time-step:** various in the different states, but typically $\sim 10^{-4}$.

**Data are stored separated by ($\Delta t$):** various at different states, but typically 1.

**Total time range ($t_{min}$ to $t_{max}$):** depends on the simulation.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $0 \leq x \leq 2\pi$, $-1 \leq y \leq 1$.

**Set of coefficients or non-dimensional parameters evaluated:** Reynold number $Re=1000$, Weissenberg number $Wi = 50$, $\beta =0.9$, $\epsilon=2.10^{-6}$, $L_{max}=70$.

**Approximate time to generate the data:** 3 months to generate all the data. It takes typically 1 day to generate $\sim 50$ snapshots.

**Hardware used to generate the data:** typically 32 or 64 cores.

> What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** The phenomena of interest in the data is: (i) chaotic dynamics in viscoelastic flows in EIT and CAR. Also note that they are separate states. (ii) multistability for the same set of parameters, the flow has four different behaviours depending on the initial conditions.

**How to evaluate a new simulator operating in this space:**
A new simulator would need to capture EIT/CAR adequately for a physically relevant parameter range.
"""

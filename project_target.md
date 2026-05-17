1. 请你完成下面的大作业，搭建一个评测框架，然后生成并优化TSMM算子，把各个版本的算子单独存储在cpp文件中，把评测脚本和算法实现分隔开，算法文件中只保留预处理函数和计算函数，降低代码耦合度。评测时需要与MKL库对比验证正确性。
2. 测评时先预热10轮，再运行20次取平均，最终指标是各个任务的GFLOPS；另外计算相对于MKL库中gemm算子的加速比，然后取几何平均，作为参考。
3. 搭建一个网页展示，实时展示测评结果，注意目标机器上不能使用docker。
4. 你先在本机生成算子，但是最终目标是优化在指定机器上的性能，后续我将手动在目标机器上进行性能评测与分析，我会把结果反馈给你，你再根据结果进一步优化。
5. 在本机可以直接运行测试，但是在目标机器需要使用slurm提交任务，使用4个NUMA node的所有CPU并行计算。

## 作业要求：

Course Project: TSMM Multiplication optimization
- 课程大作业：Tall-Skinny Matrix Multiplication (TSMM) 优化
- 目标：通过优化瘦高矩阵乘法（TSMM），掌握性能分析方法与工具、并行编程技术、访存优化技术等，锻炼解决实际问题的综合能力
- TSMM定义：
  $$
  C = A^T \times B,\ A \in R^{k \times m}, B \in R^{k \times n}, C \in R^{m \times n}
  $$
  required : $(m,n,k) = (4000,16000,128),(8,16,16000),(32,16000,16),(144,144,144)$
  optional : $(m,n,k) = (16,12344,16),(4,64,606841),(442,193,11),(40,1127228,40)$
- 假定：A、B、C均是稠密矩阵，双精度（60分）
- 实验平台：统一提供Intel平台

Course Project: TSMTTSM Multiplication optimization
- 作业要求：
- 阶段一：基础实现与分析
  - 串行实现：用C/C++编写TSMM的串行版本，支持行主序/列主序存储
  - 与库函数MKL/openBLAS等对比
  - 用性能模型、prof工具等分析性能及其优化的潜在空间
- 阶段二：并行优化
  - 多线程优化（OpenMP），设计并行分块策略，对比不同调度策略的性能差异，分析扩展性等
  - 访存模式优化，设计blocking等策略提升访存效率，并给出理论分析
  - 内核汇编优化（可选）
  - 最终优化结果与库函数MKL/openBLAS等对比
- 提交要求：
  - 代码：完整可编译的C/C++代码（30%，正确性&性能）
  - 报告：PDF/WORD格式，至少包含上述阶段内容（40%，完整性&创新性）
  - 答辩：第19周大作业答辩，分组制作PPT展示优化方法和结果分析等（30%）
  - 分组：自由组队，每组7~8人。代码&报告每个人都要单独提交一份，答辩ppt只需要小组提交一份。

## 目标CPU信息

```
[t6s008866@ln2%bscc-t6 ~]$ srun lscpu
srun: job 32437165 queued and waiting for resources
srun: job 32437165 has been allocated resources
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                96
On-line CPU(s) list:   0-95
Thread(s) per core:    1
Core(s) per socket:    48
Socket(s):             2
NUMA node(s):          4
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel(R) Xeon(R) Platinum 9242 CPU @ 2.30GHz
Stepping:              7
CPU MHz:               3099.890
CPU max MHz:           3800.0000
CPU min MHz:           1000.0000
BogoMIPS:              4600.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              1024K
L3 cache:              36608K
NUMA node0 CPU(s):     0-23
NUMA node1 CPU(s):     24-47
NUMA node2 CPU(s):     48-71
NUMA node3 CPU(s):     72-95
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 invpcid_single ssbd mba rsb_ctxsw ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req pku ospke avx512_vnni md_clear spec_ctrl intel_stibp flush_l1d arch_capabilities
```

## 环境配置

集群上可用软件列表：

```
[t6s008866@ln2%bscc-t6 ~]$ module avail
--------------------------------- /public5/soft/modulefiles ---------------------------------
2018.09.60072_bundle                                libxc/4.3.4-gcc9.3               
abacus/3.1.1-ips20-libxc5.1.7-pytorch1.12.1         libxc/4.3.4-icc18                
ACN/latest                                          libxc/5.1.0-mpiicc17             
ACN/test                                            libxc/5.2.3-icc17                
airss/0.9.1-gnu                                     libxsmm/1.15-icc17               
aviso-fes/2.9.3-ips17                               libxsmm/master                   
bagel/1.2.2-gcc10.2.0-ips18                         libxsmm/master-icc17             
bbmap/38.96                                         llvm/3.9.1                       
blas/3.8.0                                          loki/0.1.7-para                  
blas/3.8.0-mpiifort                                 mesa/13.0.6-thc                  
blast+/2.13.0                                       metabat/2.12.1                   
boost/155-intel17                                   metis/5.1-para                   
boost/171-oneAPI.2022.1-py3.9                       metis/5.1.0                      
boost/174-all-gcc9.3                                miniforge/24.11                  
boost/174-gcc-python                                mpfr/4.0.2                       
boost/174-para                                      mpi/hpcx/2.4.1-intel20           
boost/176-gcc                                       mpi/intel/17.0.5                 
CalculiX-Adapter/master-gcc7.3.0                    mpi/intel/17.0.5-cjj             
cdo/1.9.10                                          mpi/intel/17.0.7-thc             
cdo/1.9.10-ips17                                    mpi/intel/18.0.2-thc             
cgal/4.7-para                                       mpi/intel/19.3.0                 
cgns/3.4.0-hdf5-intel18-jl                          mpi/intel/20.0.4                 
cgns/3.4.0-new                                      mpi/intel/2018.4                 
cgns/4.3.0-ips17                                    mpi/intel/2021.1                 
cmake/3.9.0-zyq                                     mpi/intel/2021.2                 
cmake/3.17.0-gcc                                    mpi/intel/2022.1                 
cmake/3.21.2                                        mpi/oneAPI/2021.2                
cmake/3.22.0                                        mpi/oneAPI/2022.1                
cmake/4.2.0                                         mpi/openmpi/1.6.5                
cp2k/8.1-oneAPI.2022.1-regtest                      mpi/openmpi/3.1.2-gcc-test       
cp2k/9.1-ips20-all                                  mpi/openmpi/3.1.4-gcc            
cp2k/9.1-libxc-libint-libxsmm-plumed-icc17          mpi/openmpi/3.1.6-gcc-9.3.0      
cp2k/9.1-oneAPI.2022.1-plumed                       mpi/openmpi/4.1.1-gcc            
cp2k/2022-ips20-all                                 mpi/openmpi/4.1.1-gcc9.3.0       
cp2k/2023                                           mpi/openmpi/4.1.1-intel-17       
cp2k/2023.2-wuxm                                    mpi/openmpi/4.1.1-ips20          
cp2k/2024.1-lijia                                   mpi/openmpi/4.1.5-gcc12.2        
cp2k/2024.1-plumed                                  mpi/openmpi/5.0.8-gcc10.2.0      
cp2k/intel17/5.1-zyq                                mpich/3.0.4-gcc9.3               
cp2k/intel17/6.1-libxc-libint-zyq                   mpich/3.0.4-icc                  
cp2k/intel17/7.1-libxc-libint                       mpich/3.1.4-gcc4.9.2             
cp2k/intel17/7.1-libxc-plumed-zw                    mpich/3.1.4-gcc8.1.0             
cp2k/intel20/8.1-libxc-plumed-lcc                   mpich/3.2-gcc9.3                 
cp2k/intel20/8.1-plumed-libxc-libint-ztt            mpich/3.4.2                      
cp2k/intel20/8.1-plumed-libxc-libvori               namd/intel17/2.12-zyq            
cp2k/intell18/7.1-plumed-icc18                      ncl/6.6.2-new                    
cp2k/intell18/8.1-libinit-libxc-elpa-libxsmm-icc18  ncl/6.6.2-nodap                  
cp2k/intell18/8.2-libinit-libxc-libxsmm-icc18       nco/5.0.5-antlr-ips17            
dealii/9.3.3-ips18-trilinos-petsc-p4est             ncview/2.1.7                     
eccodes/2.21.0                                      ncview/2.1.7-new                 
eigen/3.3.8-para                                    netcdf-fortran/4.6.1             
eigen/3.4.0                                         netcdf/3.6.3-zyq                 
ElATools/1.7.3-ips17                                netcdf/4.1.3-para                
elpa/2016.05.004                                    netcdf/4.4.1-icc17               
elpa/2018.11.001                                    netcdf/4.4.1-icc21-wrf           
elpa/2019.11.001                                    netcdf/4.4.1-parallel-icc17      
espresso/4.2.0-oneAPI.2022.1-py3.9                  netcdf/4.4.1-parallel-icc20      
FFmpeg/n3.0-oneAPI.2022.1                           netcdf/4.6.3-intel20             
fftw/3.3.4-gcc-4.8.5                                netcdf/4.7.1-ips17-fortran4.5.2  
fftw/3.3.8-fenggl                                   netcdf/4.7.1-ips17-gnu           
fftw/3.3.8-gcc9.3                                   netcdf/4.7.1-parallel            
fftw/3.3.8-intel                                    netcdf/4.7.2-intel20             
fftw/3.3.8-intel-shared                             octave/7.3.0                     
fftw/3.3.8-mpi                                      octopus/10.4-intel17             
FreeFem/4.15                                        octopus/11.1-intel17             
g95/0.94                                            oneAPI/2022.1                    
gatk/4.1.4                                          openblas/0.3.17-ips18            
gatk/4.2.5                                          openmpi/3.1.4-icc17-orca         
gcc/4.9.2                                           openmpi/3.1.6-gcc                
gcc/7.3.0                                           openmpi/3.1.6-intel17            
gcc/8.1.0                                           openmpi/3.1.6-orca               
gcc/9.1.0-fenggl                                    openmpi/4.1.1-orca               
gcc/9.3.0                                           opensm/3.3.21                    
gcc/9.3.0-new                                       OPERA-MS/0.9.0                   
gcc/10.2.0                                          orca/5.0.1-shared-openmpi-4.1.1  
gcc/11.2                                            p7zip/9.20.1                     
gcc/12.2                                            p7zip/16.02                      
gcc/medea                                           PAINTOR/3.0-ips17                
gdal/3.1.4                                          paraview/5.8.0                   
geos/3.8.2                                          paraview/5.9.1                   
gklib/1.1                                           parmetis/4.0.3                   
gmp/6.3.0                                           pcre/8.42                        
gnuplot/5.2.8                                       perl/5.32.1                      
googletest/1.14.0                                   petsc/3.9.4                      
grace/5.1.25-ips17                                  petsc/3.9.4-intel18              
grads/2.0.2                                         petsc/3.11.4-intel18-hdf5-ls     
graphviz/3.0-para                                   PHengLEI/PHengLEI                
gromacs/4.6.7                                       plumed/2.4.2-icc                 
gromacs/4.6.7-fftw334                               plumed/2.5.4                     
gromacs/4.6.7-gcc                                   plumed/2.6.1                     
gromacs/4.6.7-mpicc                                 plumed/2.7.3-oneAPI.2022.1       
gromacs/5.1.5                                       pnetcdf/1.11.1-icc17             
gromacs/5.1.5-new                                   pnetcdf/1.12.1                   
gromacs/2018.1-plumed-double                        pnetcdf/1.12.1-intel20           
gromacs/2018.1-plumed-single                        pnetcdf/1.12.1-intel21-wrf       
gromacs/2018.4                                      pnetcdf/1.12.1-ips17             
gromacs/2018.4-AVX_512                              pnetcdf/1.12.1-ips17-gnu         
gromacs/2019.6-plumed-ips17                         precice/2.5.0-ips17              
gromacs/2020.1                                      Prodigal/2.6.3                   
gromacs/2020.6-intel-2021                           proj/6.2.1                       
gromacs/2021.1-cp2k                                 pwmat/pwmat-20231122             
gromacs/2021.2                                      python/2.7.1-tzm                 
gromacs/2021.3-avx512-intel-20.0.4                  python/3.8.6                     
gromacs/2022.2-oneAPI.2022.1                        python/3.9.6                     
gromacs/:                                           python/3.9.6-cjj                 
gsl/2.5-cjj                                         python/3.9.6-para                
gsl/2.6                                             qe/6.2                           
gts/0.7.6-para                                      qe/6.4.1-lcc                     
guile/3.0.8-gcc8.1.0                                qe/6.5-lcc                       
hdf5/1.8.13-icc17                                   qe/6.6-ips17                     
hdf5/1.8.13-parallel-icc17                          qe/6.7.0-oneAPI.2022.1           
hdf5/1.8.13-parallel-icc18                          qe/7.0.0-oneAPI.2022.1           
hdf5/1.8.13-parallel-icc20                          qe/7.3-2022.1-zjm                
hdf5/1.8.18                                         qt/5.13.0                        
hdf5/1.8.18-icc18                                   qt/5.13.2                        
hdf5/1.8.18-intel20                                 R/3.6.1                          
hdf5/1.10.1                                         R/4.0.2                          
hdf5/1.10.4-intel20                                 R/4.0.3                          
hdf5/1.10.4-ips17-gnu                               R/4.0.3-Seurat                   
hdf5/1.10.5-icc21                                   R/4.0.5                          
hdf5/1.10.5-ips17                                   R/4.1.2-ips17                    
hdf5/1.10.6-intel20                                 R/4.2.2                          
hdf5/1.10.6-openmpi-3.1.6-gcc-9.3.0                 rar/3.8.0                        
hdf5/1.12.2-openmpi4.1.1                            rar/6.0.1                        
HiFiLES-solver/latest                               s3/mc                            
hmmer/3.3.2                                         s3/rclone                        
hpcx/2.4.1                                          samtools/1.15                    
hpcx/2.4.1-gnu10.2                                  scalapack/2.1.0                  
hypre/hypre-2.11.2                                  script/wxl                       
intel/15.0.6                                        siesta/4.1.5-ips17-psml-gridxc   
intel/17.0.5                                        siesta/max-1.3.0-para            
intel/17.0.5-cjj                                    singularity/3.7.0                
intel/17.0.7-thc                                    singularity/3.7.4                
intel/18.0.2-thc                                    singularity/3.9.9                
intel/18.0.4                                        slurm/22.05.10                   
intel/19.3.0                                        slurm/stable                     
intel/20.0.4                                        spglib/1.11.2.1                  
intel/2021.1                                        swig/3.0.0                       
intel/2021.2                                        swig/4.1.0                       
intel/2022.1                                        szip/2.1.1                       
jasper/1.9                                          szip/2.1.1-ls                    
jasper/1.9.1-hsh-17                                 ucx/1.15.0                       
jasper/1.9.1-hsh-18                                 udunits/2.2.26                   
jasper/1.9.1-hsh-20                                 v_sim/3.7.2-ips17                
java/1.8.0_221                                      vaspkit/1.4.1-Pro-PARATERA       
lammps/intel17/3Mar20                               vaspkit/1.5.0-Pro-PARATERA       
lammps/intel17/11Aug2017-ips17                      vmd/1.9.2                        
lammps/intel17/22Aug18                              vtk/8.0.2-zyq                    
lammps/intel17/27May2021-basic-avx512               vtk/8.2.0-para                   
lammps/intel17/29Oct20                              vtk/9.1.0                        
lammps/intel17/29Sep2021-meam-voronoi               wannier90/intel17/2.1.0          
lammps/intel17/30Apr19-ips17-fix                    wannier90/intel17/3.1.0          
lammps/intel2022.1/22Dec2022                        wannier_tools/intel17-lmy        
lammps/oneAPI.2022.1/7Feb2024-no_lib                wps/3.9                          
lammps/oneAPI.2022.1/15Jun2023-voronoi-lepton       wps/4.0-ips17                    
lammps/oneAPI.2022.1/29Oct2020-no_lib               wps/4.2                          
lammps/oneAPI.2022.1/stable_2Aug2023-para           wps/4.3                          
lapack/3.9.0                                        wps/4.4-ips17                    
lapack/3.9.0-gcc9.3.0                               wrf/3.9                          
libgridxc/0.9.6-para                                wrf/3.9.1                        
libint/1.1.5-icc17-zyq                              wrf/4.0-ips17                    
libint/2.6.0-cp2k-icc17                             wrf/4.2                          
libint/2.6.0-icc18                                  wrf/4.3                          
libint/libint-v2.6.0-cp2k-lmax-4                    wrf/4.4-ips17                    
libpng/1.6.37                                       wrf/4.4-ips17-hydro5.2           
libpng/1.6.37-hsh-17                                wxl/0.1                          
libpng/1.6.37-hsh-18                                xmlf90/1.5.4-para                
libpng/1.6.37-hsh-20                                xorriso/1.5.4                    
libpsml/1.1.8-para                                  yade/2022.01-para                
libxc/2.2.3                                         yade/trunk-2021.01a-ips17        
libxc/3.0.0-para                                    yasm/1.3.0-ips17                 
libxc/4.0.5-icc17-zyq                               zlib/1.2.7                       
libxc/4.0.5-intel20                                 zlib/1.2.7-21                    
libxc/4.0.5-mpiicc17 
```

默认已加载的软件：

```
[t6s008866@ln2%bscc-t6 ~]$ module list
Currently Loaded Modulefiles:
 1) gcc/10.2.0   2) cmake/3.21.2   3) python/3.8.6  
```
## Levenberg-Marquardt Optimization using Eigen

Check out the [blog post](https://medium.com/@sarvagya.vaish/levenberg-marquardt-optimization-part-2-5a71f7db27a0)

## Build and Run

```
mkdir build
cd build
# Depending on how you set up your installation of eigen ...
# An example follows:
# cmake -DEIGEN3_INCLUDE_DIR=/home/peno/install/eigen/include/eigen3/ ..
cmake ..
make
./LMOptimize
```

Output:
```
LM optimization status: 2
Optimization results
    a: -1.99839
    b: 50.0387
    c: 7.83367
```


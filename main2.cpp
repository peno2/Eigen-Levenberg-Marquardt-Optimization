#include <iostream>
#include <Eigen/Dense>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

namespace {
double warp2(const Eigen::VectorXd &x) { return  fabs(x(0) - 16) + fabs(x(1) - 8); }
double warp3(const Eigen::VectorXd &x) { return  fabs(x(0) - 15) + fabs(x(1) - 7) + fabs(x(2) + 9); }
}

constexpr int noOfParams = 2;
// Generic functor
template<typename Scalar_, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor {
  using Scalar = Scalar_;
  enum {
	InputsAtCompileTime = NX,
	ValuesAtCompileTime = NY
  };
  using  InputType = Eigen::Matrix<Scalar, InputsAtCompileTime, 1>;
  using  ValueType = Eigen::Matrix<Scalar, ValuesAtCompileTime, 1>;
  using  JacobianType = Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;

  //Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

private:
  int m_inputs, m_values;
};

struct my_functor : Functor<double> {
  my_functor(): Functor<double>(noOfParams, noOfParams) {}
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    fvec(0) = warp2(x);
    fvec(1) = 0;

    return 0;
  }
};


int main(int argc, char *argv[]) {
  Eigen::VectorXd x(noOfParams);
  std::cout << "x:\n" << x << std::endl;

  const my_functor functor;
  Eigen::NumericalDiff<my_functor> numDiff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>,double> lm(numDiff);
  lm.parameters.maxfev = 2000;
  lm.parameters.xtol = 1.0e-10;
  std::cout << "Maxfev: " << lm.parameters.maxfev << std::endl;

  const int ret = lm.minimize(x);
  std::cout << "Iter : " << lm.iter << "\n"
	    << "Ret: " << ret << "\n"
	    << "x that minimizes the function:\n" << x << "\n";
  return 0;
}

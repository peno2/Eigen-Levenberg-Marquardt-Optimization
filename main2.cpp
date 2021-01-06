#include <iostream>
#include <Eigen/Dense>
#include <cassert>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

// From: https://stackoverflow.com/questions/59747377/custom-scalar-type-in-eigen
class MyDouble {
    public:
        double value;
        MyDouble() : value() {};
        MyDouble(double val) : value(val) {};

        template<typename T>
            MyDouble& operator=(T rhs) {
                value = static_cast<double>(rhs);
                return *this;
            }

        template<typename T>
            MyDouble& operator+=(T rhs) {
                value = static_cast<double>(value + rhs.value);
                return *this;
            }

        template<typename T>
            MyDouble& operator-=(const T &rhs) {
                value = static_cast<double>(value - rhs.value);
                return *this;
            }

        template<typename T>
            MyDouble& operator*=(T rhs) {
                value = static_cast<double>(value * rhs.value);
                return *this;
            }

        template<typename T>
            MyDouble& operator/=(T rhs) {
                value = static_cast<double>(value / rhs.value);
                return *this;
            }

        MyDouble operator-() const {
            return -value;
        }

        friend std::ostream& operator<<(std::ostream& out, const MyDouble& val) {
            out << val.value << " m";
            return out;
        }

        operator double() const {
            return value;
        }
};

#define OVERLOAD_OPERATOR(op,ret) ret operator op(const MyDouble &lhs, const MyDouble &rhs) { \
        return lhs.value op rhs.value; \
    }

#define OVERLOAD_OPERATOR_D0(op,ret) ret operator op(const double &lhs, const MyDouble &rhs) { \
        return lhs op rhs.value; \
    }

#define OVERLOAD_OPERATOR_D(op,ret) ret operator op(const MyDouble &lhs, const double &rhs) { \
        return lhs.value op rhs; \
    }

#define OVERLOAD_OPERATOR_I0(op,ret) ret operator op(const int &lhs, const MyDouble &rhs) { \
        return lhs op rhs.value; \
    }

#define OVERLOAD_OPERATOR_I1(op,ret) ret operator op(const MyDouble &lhs, const int &rhs) { \
        return lhs.value op rhs; \
    }

OVERLOAD_OPERATOR(+, MyDouble)
OVERLOAD_OPERATOR(-, MyDouble)
OVERLOAD_OPERATOR(*, MyDouble)
OVERLOAD_OPERATOR(/, MyDouble)

OVERLOAD_OPERATOR(>, bool)
OVERLOAD_OPERATOR(<, bool)
OVERLOAD_OPERATOR(>=, bool)
OVERLOAD_OPERATOR(<=, bool)
OVERLOAD_OPERATOR(==, bool)
OVERLOAD_OPERATOR(!=, bool)

OVERLOAD_OPERATOR_D(+, MyDouble)
OVERLOAD_OPERATOR_D(-, MyDouble)
OVERLOAD_OPERATOR_D(*, MyDouble)
OVERLOAD_OPERATOR_D(/, MyDouble)

OVERLOAD_OPERATOR_D(>, bool)
OVERLOAD_OPERATOR_D(<, bool)
OVERLOAD_OPERATOR_D(>=, bool)
OVERLOAD_OPERATOR_D(<=, bool)
OVERLOAD_OPERATOR_D(==, bool)
OVERLOAD_OPERATOR_D(!=, bool)

OVERLOAD_OPERATOR_I0(*, MyDouble)
OVERLOAD_OPERATOR_D0(-, MyDouble)
OVERLOAD_OPERATOR_I1(!=, bool)

MyDouble sqrt(MyDouble val) {
    return std::sqrt(val.value);
}
MyDouble abs(MyDouble val) {
    return std::abs(val.value);
}
MyDouble abs2(MyDouble val) {
    return val * val;
}
bool isfinite(const MyDouble &) { return true; }

namespace Eigen {
    template<> struct NumTraits<MyDouble>
        : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
        {
            typedef MyDouble Real;
            typedef MyDouble NonInteger;
            typedef MyDouble Nested;
            enum {
                IsComplex = 0,
                IsInteger = 0,
                IsSigned = 1,
                RequireInitialization = 0,
                ReadCost = 1,
                AddCost = 3,
                MulCost = 3
            };
	  inline static MyDouble epsilon() { return NumTraits<double>::epsilon(); }
        };

    template<typename BinaryOp>
    struct ScalarBinaryOpTraits<MyDouble,double,BinaryOp> { typedef MyDouble ReturnType;  };

    template<typename BinaryOp>
    struct ScalarBinaryOpTraits<double,MyDouble,BinaryOp> { typedef MyDouble ReturnType;  };
}

using MyType = MyDouble;
using MyVec = Eigen::Matrix<MyType, Eigen::Dynamic, 1>;

namespace {
  double warp2(const MyVec &x) { return  fabs(x(0) - MyType{16.}) + fabs(x(1) - MyType{8.}); }
  double warp3(const MyVec &x) { return  fabs(x(0) - MyType{15}) + fabs(x(1) - MyType{7}) + fabs(x(2) + MyType{9}); }
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
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {
    assert(m_inputs == 1);
  }

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

private:
  int m_inputs, m_values;
};

struct my_functor : Functor<MyType> {
  my_functor(): Functor<MyType>(1, noOfParams) {}
  int operator()(const MyVec &x, MyVec &fvec) const {
    fvec(0) = warp2(x);

    return 0;
  }
};


int main(int argc, char *argv[]) {
  MyVec x(noOfParams);
  std::cout << "Inital value of x:\n" << x << std::endl;

  const my_functor functor;
  const MyType minEpsilon = 1;
  Eigen::NumericalDiff<my_functor> numDiff(functor, minEpsilon);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>, MyType> lm(numDiff);
  lm.parameters.maxfev = 2000;
  lm.parameters.xtol = 1.0e-10;
  std::cout << "Maxfev: " << lm.parameters.maxfev << std::endl;

  const int ret = lm.minimize(x);
  std::cout << "Iter : " << lm.iter << "\n"
	    << "Ret: " << ret << "\n"
	    << "x that minimizes the function:\n" << x << "\n";
  return 0;
}
